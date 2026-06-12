import os
import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from config.stereo_human_config import ConfigStereoHuman as config
from lib.network import VDAGaussianModel
from lib.GaussianRender import pts2render
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from lib.utils import depth2pc

cv2.setNumThreads(0)
torch.set_float32_matmul_precision('high')

# GPU 텐서 레벨 패딩 함수
def pad_to_square_tensor(tensor):
    h, w = tensor.shape[2], tensor.shape[3]
    max_side = max(h, w)
    pad_t = (max_side - h) // 2
    pad_b = max_side - h - pad_t
    pad_l = (max_side - w) // 2
    pad_r = max_side - w - pad_l
    return F.pad(tensor, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0.0)

def run_video_inference(vid_path, ckpt_path, ref_intr_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_name = Path(vid_path).stem
    
    cfg_wrapper = config()
    cfg_wrapper.load("config/stage2.yaml")
    cfg = cfg_wrapper.get_cfg()
    cfg.defrost()
    cfg.dataset.src_res = 504
    cfg.freeze()
    
    model = VDAGaussianModel(cfg, with_gs_render=True)
    model.cuda()
    model.eval()
    
    print("Loading Model Checkpoint...")
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['network'], strict=True)

    print("Loading RVM Model...")
    rvm = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda().eval()

    # 카메라 파라미터 로드
    raw_intr = np.load(ref_intr_path)
    ref_extr_path = ref_intr_path.replace('intr.npy', 'extr.npy')
    extr_np = np.load(ref_extr_path)
    if extr_np.shape == (3, 4):
        extr_4x4 = np.eye(4, dtype=np.float32)
        extr_4x4[:3, :] = extr_np
        extr_np = extr_4x4
        
    intr_504 = raw_intr.copy()
    intr_tensor_504 = torch.from_numpy(intr_504).float().unsqueeze(0).cuda()
    extr_tensor = torch.from_numpy(extr_np).float().unsqueeze(0).cuda()

    render_res = 504
    fovx = focal2fov(intr_504[0, 0], render_res)
    fovy = focal2fov(intr_504[1, 1], render_res)
    proj_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, K=intr_504, h=render_res, w=render_res).transpose(0, 1).cuda()
    bg_color = getattr(cfg.dataset, 'bg_color', [0.0, 0.0, 0.0])
    base_extr = extr_tensor[0]

    # 비디오 읽기 설정
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0

    print(f"Processing Video: {total_frames} frames @ {fps} FPS")

    # 💡 RVM용 RNN Hidden State 초기화
    rec = [None] * 4 
    frames_out = []

    t_pipeline_start = time.perf_counter()

    for frame_idx in tqdm(range(total_frames), desc="Video Inference (Baseline)"):
        ret, frame = cap.read()
        if not ret: break

        # 1. BGR -> RGB 변환 및 텐서화
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor_raw = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0

        with torch.no_grad():
            b, c, h, w = img_tensor_raw.shape
            ds_ratio = min(1.0, 512.0 / max(h, w)) 
            # 💡 이전 프레임의 상태값(*rec)을 넘겨주어 시간적 일관성 확보
            fgr, pha, *rec = rvm(img_tensor_raw, *rec, downsample_ratio=ds_ratio)

        mask_tensor_raw = (pha > 0.15).float() 
        img_clean_tensor = img_tensor_raw * pha 

        padded_img = pad_to_square_tensor(img_clean_tensor)
        padded_mask = pad_to_square_tensor(mask_tensor_raw)

        img_tensor = F.interpolate(padded_img, size=(render_res, render_res), mode='area')
        mask_tensor = F.interpolate(padded_mask, size=(render_res, render_res), mode='nearest')
        img_gps = img_tensor * 2.0 - 1.0

        # 2. 네트워크 추론
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True):
                img_feat = model.unet_extractor(img_gps)
            
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                depth_seq, _ = model.vda_model(img_tensor.unsqueeze(1))
            
            depth_pred = depth_seq.squeeze(1).unsqueeze(1).float()
            rot, scale, opacity = model.gs_parm_regresser(img_gps, depth_pred, img_feat)

        bs = img_tensor.shape[0]
        data = {'view_0': {'img': img_tensor, 'mask': mask_tensor, 'intr': intr_tensor_504, 'extr': extr_tensor}}
        data['view_0']['depth'] = depth_pred
        data['view_0']['xyz'] = depth2pc(depth_pred, extr_tensor, intr_tensor_504).view(bs, -1, 3)
        
        valid_mask = (depth_pred > 0.1).view(bs, -1)
        depth_flat = depth_pred.view(valid_mask.shape)
        mask_flat = mask_tensor.view(valid_mask.shape)
        data['view_0']['pts_valid'] = valid_mask & (depth_flat > 0.3) & (mask_flat > 0.5)
        
        data['view_0']['rot_maps'] = rot.view(bs, 4, render_res, render_res)
        data['view_0']['scale_maps'] = scale.view(bs, 3, render_res, render_res)
        data['view_0']['opacity_maps'] = opacity.view(bs, 1, render_res, render_res)

        # 3. 렌더링 (영상의 진행도에 따라 카메라가 좌우로 패닝)
        progress = frame_idx / float(total_frames)
        dx = np.sin(progress * 2 * np.pi) * 0.30 
        
        novel_extr = base_extr.clone()
        novel_extr[0, 3] -= dx 
        R = novel_extr[:3, :3].T
        T = novel_extr[:3, 3]
        
        world_view_transform = torch.tensor(getWorld2View2(R.cpu().numpy(), T.cpu().numpy(), np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).float().cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(proj_matrix.unsqueeze(0))).squeeze(0)
        cam_center = world_view_transform.inverse()[3, :3]
        
        data['novel_view'] = {
            'width': torch.tensor([render_res], dtype=torch.int32).cuda(),
            'height': torch.tensor([render_res], dtype=torch.int32).cuda(),
            'FovX': torch.tensor([fovx], dtype=torch.float32).cuda(),
            'FovY': torch.tensor([fovy], dtype=torch.float32).cuda(),
            'world_view_transform': world_view_transform.unsqueeze(0),
            'full_proj_transform': full_proj_transform.unsqueeze(0),
            'camera_center': cam_center.unsqueeze(0)
        }
        
        with torch.no_grad():
            render_out = pts2render({'lmain': data['view_0'], 'novel_view': data['novel_view']}, bg_color=bg_color, is_train=False)
            
        render_img_504 = render_out['novel_view']['img_pred'][0].detach().permute(1, 2, 0).cpu().numpy()
        render_img_1008 = cv2.resize(render_img_504, (1008, 1008), interpolation=cv2.INTER_LANCZOS4)
        frames_out.append(np.clip(render_img_1008 * 255.0, 0, 255).astype(np.uint8))

    cap.release()
    t_pipeline_end = time.perf_counter()

    t_io_start = time.perf_counter()
    out_mp4 = os.path.join(out_dir, f"{out_name}_Baseline_Video.mp4")
    imageio.mimsave(out_mp4, frames_out, fps=fps, macro_block_size=1)
    t_io_end = time.perf_counter()

    print("\n===================================")
    print(" [Baseline Video Inference Profiling]")
    print(f" - Total Frames Processed: {len(frames_out)}")
    print(" - Total Processing Time : {:.2f} s".format((t_pipeline_end - t_pipeline_start)))
    print(" - Average FPS (Infer)   : {:.2f} FPS".format(len(frames_out) / (t_pipeline_end - t_pipeline_start)))
    print(" - Video Saving Time     : {:.2f} s".format((t_io_end - t_io_start)))
    print("===================================\n")

if __name__ == "__main__":
    VID_PATH = "input_video.mp4" # 💡 여기에 비디오 경로 입력
    CKPT_PATH = "experiments/VDA_GPS_0529_Finetune/ckpt/VDA_GPS_0529_Finetune_final.pth"
    REF_INTR_PATH = "../thuman_120cm_render_data_mono/mono_uniform_504/val/parm/0000_000/0_intr.npy"
    OUT_DIR = "baseline_results"

    run_video_inference(VID_PATH, CKPT_PATH, REF_INTR_PATH, OUT_DIR)