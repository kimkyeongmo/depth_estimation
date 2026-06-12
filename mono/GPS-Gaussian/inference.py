import os
import time
import torch
import cv2
import numpy as np
import math
import imageio
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from rembg import remove

from config.stereo_human_config import ConfigStereoHuman as config
from lib.network import VDAGaussianModel
from lib.GaussianRender import pts2render
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from lib.utils import depth2pc

cv2.setNumThreads(0)
torch.set_float32_matmul_precision('high')


import warnings
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

def pad_to_square(img_np, is_mask=False):
    h, w = img_np.shape[:2]
    max_side = max(h, w)
    pad_t = (max_side - h) // 2
    bottom = max_side - h - pad_t
    pad_l = (max_side - w) // 2
    right = max_side - w - pad_l
    pad_value = 0 if is_mask else [0, 0, 0]
    padded = cv2.copyMakeBorder(img_np, pad_t, bottom, pad_l, right, cv2.BORDER_CONSTANT, value=pad_value)
    return padded, pad_t, pad_l, max_side

def run_inference(img_path, ckpt_path, ref_intr_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_name = Path(img_path).stem
    
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
    
    t_pipeline_start = time.perf_counter()
    
    print("Processing Image...")
    t_prep_start = time.perf_counter()
    pil_img = Image.open(img_path).convert("RGB")
    processed_pil = remove(pil_img) 
    
    img_clean = np.array(processed_pil.convert("RGB"))
    mask = np.array(processed_pil.split()[3]) 
    mask = (mask > 128).astype(np.uint8) 
    
    padded_img, pad_t, pad_l, S = pad_to_square(img_clean, is_mask=False)
    padded_mask, _, _, _ = pad_to_square(mask, is_mask=True)
    
    infer_res = 504
    img_504 = cv2.resize(padded_img, (infer_res, infer_res), interpolation=cv2.INTER_AREA)
    mask_504 = cv2.resize(padded_mask, (infer_res, infer_res), interpolation=cv2.INTER_NEAREST)
    
    img_tensor = torch.from_numpy(img_504).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    mask_tensor = torch.from_numpy(mask_504).float().unsqueeze(0).unsqueeze(0)
    
    img_tensor = img_tensor.cuda()
    mask_tensor = mask_tensor.cuda()

    raw_intr = np.load(ref_intr_path)
    ref_extr_path = ref_intr_path.replace('intr.npy', 'extr.npy')
    extr_np = np.load(ref_extr_path)
    if extr_np.shape == (3, 4):
        extr_4x4 = np.eye(4, dtype=np.float32)
        extr_4x4[:3, :] = extr_np
        extr_np = extr_4x4
        
    intr_tensor_504 = torch.from_numpy(raw_intr.copy()).float().unsqueeze(0).cuda()
    extr_tensor = torch.from_numpy(extr_np).float().unsqueeze(0).cuda()
    t_prep_end = time.perf_counter()

    print("Running GPU Warm-up...")
    t_warmup_start = time.perf_counter()
    img_gps = img_tensor * 2.0 - 1.0
    with torch.no_grad():
        _ = model.unet_extractor(img_gps)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            _, _ = model.vda_model(img_tensor.unsqueeze(1))
    torch.cuda.synchronize()
    t_warmup_end = time.perf_counter()
    time_warmup = (t_warmup_end - t_warmup_start) * 1000

    print("Benchmarking Sub-modules...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        # 1. U-Net Timing
        start_event.record()
        with torch.amp.autocast('cuda', enabled=True):
            img_feat = model.unet_extractor(img_gps)
        end_event.record()
        torch.cuda.synchronize()
        time_unet = start_event.elapsed_time(end_event)

        # 2. VDA Timing
        start_event.record()
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            depth_seq, _ = model.vda_model(img_tensor.unsqueeze(1))
        end_event.record()
        torch.cuda.synchronize()
        time_vda = start_event.elapsed_time(end_event)
        
        depth_pred = depth_seq.squeeze(1).unsqueeze(1).float()

        # 3. GSRegresser Timing
        start_event.record()
        rot, scale, opacity = model.gs_parm_regresser(img_gps, depth_pred, img_feat)
        end_event.record()
        torch.cuda.synchronize()
        time_gs = start_event.elapsed_time(end_event)
        
    # 데이터 수동 조립
    bs = img_tensor.shape[0]
    data = {'view_0': {'img': img_tensor, 'mask': mask_tensor, 'intr': intr_tensor_504, 'extr': extr_tensor}}
    data['view_0']['depth'] = depth_pred
    data['view_0']['xyz'] = depth2pc(depth_pred, extr_tensor, intr_tensor_504).view(bs, -1, 3)
    data['view_0']['pts_valid'] = ((depth_pred > 0.1) & (depth_pred > 0.3) & (mask_tensor > 0.5)).view(bs, -1)
    data['view_0']['rot_maps'] = rot.view(bs, 4, 504, 504)
    data['view_0']['scale_maps'] = scale.view(bs, 3, 504, 504)
    data['view_0']['opacity_maps'] = opacity.view(bs, 1, 504, 504)

    render_res = 504
    proj_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, K=raw_intr, h=render_res, w=render_res).transpose(0, 1).cuda()
    frames = []
    steps = np.linspace(0, 2 * np.pi, 60)
    bg_color = getattr(cfg.dataset, 'bg_color', [0.0, 0.0, 0.0])
    base_extr = extr_tensor[0]

    t_render_start = time.perf_counter()
    for step in tqdm(steps, desc="Rendering & Upsampling"):
        dx = np.sin(step) * 0.30
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
            'FovX': torch.tensor([focal2fov(raw_intr[0,0], render_res)], dtype=torch.float32).cuda(),
            'FovY': torch.tensor([focal2fov(raw_intr[1,1], render_res)], dtype=torch.float32).cuda(),
            'world_view_transform': world_view_transform.unsqueeze(0),
            'full_proj_transform': full_proj_transform.unsqueeze(0),
            'camera_center': cam_center.unsqueeze(0)
        }
        
        with torch.no_grad():
            render_out = pts2render({'lmain': data['view_0'], 'novel_view': data['novel_view']}, bg_color=bg_color, is_train=False)
            
        render_img_504 = render_out['novel_view']['img_pred'][0].detach().permute(1, 2, 0).cpu().numpy()
        render_img_1008 = cv2.resize(render_img_504, (1008, 1008), interpolation=cv2.INTER_LANCZOS4)
        frames.append(np.clip(render_img_1008 * 255.0, 0, 255).astype(np.uint8))
    t_render_end = time.perf_counter()
    
    t_io_start = time.perf_counter()
    imageio.mimsave(os.path.join(out_dir, f"{out_name}_PyTorch_panning.mp4"), frames, fps=30, macro_block_size=1)
    
    sample_img = frames[15]
    out_img = os.path.join(out_dir, f"{out_name}_PyTorch_sample.jpg")
    cv2.imwrite(out_img, cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR))
    t_io_end = time.perf_counter()
    
    t_pipeline_end = time.perf_counter()

    print("\n===================================")
    print(" [End-to-End Pipeline Profiling (Baseline)]")
    print(" - Pre-processing        : {:.2f} ms".format((t_prep_end - t_prep_start) * 1000))
    print(" - Warm-up (Cold Start)  : {:.2f} ms".format(time_warmup))
    print(" - Network Inference     : {:.2f} ms".format(time_unet + time_vda + time_gs))
    print("   ├─ U-Net (PT)         : {:.2f} ms".format(time_unet))
    print("   ├─ VDA (PT BF16)      : {:.2f} ms".format(time_vda))
    print("   └─ GS Regresser (PT)  : {:.2f} ms".format(time_gs))
    print(" - Total CUDA Operation  : {:.2f} ms".format(time_warmup + time_unet + time_vda + time_gs))
    print(" - Render & Upsample (60): {:.2f} ms".format((t_render_end - t_render_start) * 1000))
    print(" - Video & Image I/O     : {:.2f} ms".format((t_io_end - t_io_start) * 1000))
    print("-----------------------------------")
    print(" - Total Pipeline Time   : {:.2f} ms".format((t_pipeline_end - t_pipeline_start) * 1000))
    print("===================================\n")

if __name__ == "__main__":
    run_inference("test.jpg", "experiments/VDA_GPS_0529_Finetune/ckpt/VDA_GPS_0529_Finetune_final.pth", 
                  "../thuman_120cm_render_data_mono/mono_uniform_504/val/parm/0000_000/0_intr.npy", "baseline_results")