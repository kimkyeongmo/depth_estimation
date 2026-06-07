import os
import sys
import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm
import tensorrt as trt

# VDA 네이티브 스트리밍 모듈 임포트
sys.path.append(os.path.abspath('../Video-Depth-Anything'))
try:
    from video_depth_anything.video_depth_stream import VideoDepthAnything as StreamingVDA
except ImportError:
    print("Error: Could not import StreamingVDA. Check your Video-Depth-Anything path.")
    exit()

from config.stereo_human_config import ConfigStereoHuman as config
from lib.network import VDAGaussianModel
from lib.GaussianRender import pts2render
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from lib.utils import depth2pc

cv2.setNumThreads(0)
torch.set_float32_matmul_precision('high')

class TRTWrapper:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.engine.get_tensor_shape(name))
                trt_dtype = self.engine.get_tensor_dtype(name)
                if trt_dtype == trt.DataType.HALF:
                    torch_dtype = torch.float16
                elif trt_dtype == trt.DataType.INT32:
                    torch_dtype = torch.int32
                elif hasattr(trt.DataType, 'BFLOAT16') and trt_dtype == trt.DataType.BFLOAT16:
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float32
                self.outputs[name] = torch.empty(shape, dtype=torch_dtype, device='cuda')
                self.context.set_tensor_address(name, self.outputs[name].data_ptr())

    def __call__(self, **inputs):
        keep_alive = []
        for name, tensor in inputs.items():
            tensor = tensor.contiguous()
            keep_alive.append(tensor)
            self.context.set_tensor_address(name, tensor.data_ptr())
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        res = {}
        for k, v in self.outputs.items():
            res[k] = v.to(torch.float32)
        return res

# 💡 GPU 텐서용 초고속 패딩 함수
def pad_to_square_tensor(tensor):
    h, w = tensor.shape[2], tensor.shape[3]
    max_side = max(h, w)
    pad_t = (max_side - h) // 2
    pad_b = max_side - h - pad_t
    pad_l = (max_side - w) // 2
    pad_r = max_side - w - pad_l
    return F.pad(tensor, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0.0)

def get_video_rotation(vid_path):
    try:
        import ffmpeg
        meta = ffmpeg.probe(vid_path)
        return int(meta['streams'][0]['tags'].get('rotate', 0))
    except:
        return 0

def run_ultimate_streaming_inference(vid_path, ckpt_path, engine_dir, ref_intr_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_name = Path(vid_path).stem

    cfg_wrapper = config()
    cfg_wrapper.load("config/stage2.yaml")
    cfg = cfg_wrapper.get_cfg()
    cfg.defrost()
    cfg.dataset.src_res = 504
    cfg.freeze()

    print("Loading PyTorch Checkpoints...")
    base_model = VDAGaussianModel(cfg, with_gs_render=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    base_model.load_state_dict(ckpt['network'], strict=True)
    vda_weights = base_model.vda_model.state_dict()
    del base_model 

    print("Initializing Stateful VDA Streamer...")
    vda_stream = StreamingVDA(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    vda_stream.load_state_dict(vda_weights, strict=True)
    vda_stream = vda_stream.cuda().eval()

    print("Loading TensorRT Engines & RVM...")
    unet_trt = TRTWrapper(os.path.join(engine_dir, "unet_extractor.engine"))
    gs_trt = TRTWrapper(os.path.join(engine_dir, "gs_regresser.engine"))
    rvm = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda().eval()

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
    target_res = 1008
    fovx = focal2fov(intr_504[0, 0], render_res)
    fovy = focal2fov(intr_504[1, 1], render_res)
    proj_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, K=intr_504, h=render_res, w=render_res).transpose(0, 1).cuda()
    bg_color = getattr(cfg.dataset, 'bg_color', [0.0, 0.0, 0.0])
    base_extr = extr_tensor[0]

    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    vid_rotation = get_video_rotation(vid_path)

    print(f"\nProcessing Video: {total_frames} frames @ {fps} FPS")

    out_mp4 = os.path.join(out_dir, out_name + "_FlickerFree_UltraFast.mp4")
    video_writer = imageio.get_writer(out_mp4, fps=fps, macro_block_size=1)

    rec = [None] * 4 
    t_pipeline_start = time.perf_counter()

    for frame_idx in tqdm(range(total_frames), desc="Ultra-Fast Streaming Inference"):
        ret, frame = cap.read()
        if not ret: break

        if vid_rotation == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif vid_rotation == 180: frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif vid_rotation == 270 or vid_rotation == -90: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor_raw = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0

        # 1. RVM 마스킹 (GPU)
        with torch.no_grad():
            b, c, h, w = img_tensor_raw.shape
            ds_ratio = min(1.0, 512.0 / max(h, w)) 
            fgr, pha, *rec = rvm(img_tensor_raw, *rec, downsample_ratio=ds_ratio)
        
        # 💡 [초고속 최적화 1] 원본 해상도(1080p 등)에서의 무거운 배열 연산을 모두 GPU 텐서로 처리
        img_clean_tensor = img_tensor_raw * pha 

        # 💡 [초고속 최적화 2] OpenCV(CPU) 대신 GPU에서 빛의 속도로 패딩 및 리사이즈 수행
        padded_img = pad_to_square_tensor(img_clean_tensor)
        padded_mask = pad_to_square_tensor(pha)

        img_tensor = F.interpolate(padded_img, size=(render_res, render_res), mode='area')
        mask_tensor = F.interpolate(padded_mask, size=(render_res, render_res), mode='nearest')
        
        # 💡 [초고속 최적화 3] 이미 504x504로 작아진 가벼운 이미지만 CPU로 넘겨 VDA 스트림 통과
        # (데이터 이동 병목 완벽 제거)
        img_504_np = (img_tensor[0].permute(1, 2, 0).detach() * 255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)

        # VDA 스트림 통과 (이미 504x504이므로 내부 리사이즈 부하도 없음)
        depth_np = vda_stream.infer_video_depth_one(img_504_np, input_size=504, device='cuda', fp32=True)

        # 💡 [초고속 최적화 4] 출력된 Depth를 CPU 패딩 없이 곧바로 GPU 텐서로 변환
        depth_pred = torch.from_numpy(depth_np).float().unsqueeze(0).unsqueeze(0).cuda()
        img_gps = img_tensor * 2.0 - 1.0

        # 3DGS 파라미터 추론 및 렌더링 (모두 GPU)
        with torch.no_grad():
            unet_out = unet_trt(input_image_gps=img_gps)
            img_feat = (unet_out['feat1'], unet_out['feat2'], unet_out['feat3'])

            gs_out = gs_trt(img_gps=img_gps, depth=depth_pred, feat1=img_feat[0], feat2=img_feat[1], feat3=img_feat[2])
            rot = gs_out['rot_maps']
            scale = gs_out['scale_maps']
            opacity = gs_out['opacity_maps']

        bs = img_tensor.shape[0]
        data = {'view_0': {'img': img_tensor, 'mask': mask_tensor, 'intr': intr_tensor_504, 'extr': extr_tensor}}
        data['view_0']['depth'] = depth_pred
        data['view_0']['xyz'] = depth2pc(depth_pred, extr_tensor, intr_tensor_504).view(bs, -1, 3)

        valid_mask = (depth_pred > 0.05).view(bs, -1)
        depth_flat = depth_pred.view(valid_mask.shape)
        mask_flat = mask_tensor.view(valid_mask.shape)

        data['view_0']['pts_valid'] = valid_mask & (mask_flat > 0.5)
        
        data['view_0']['rot_maps'] = rot.view(bs, 4, render_res, render_res)
        data['view_0']['scale_maps'] = scale.view(bs, 3, render_res, render_res)
        data['view_0']['opacity_maps'] = opacity.view(bs, 1, render_res, render_res)

        novel_extr = base_extr.clone()
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

        # GPU 고속 리사이즈
        render_tensor_504 = render_out['novel_view']['img_pred']
        render_tensor_1008 = F.interpolate(render_tensor_504, size=(target_res, target_res), mode='bicubic', align_corners=False)
        render_img_1008 = (render_tensor_1008[0].detach().permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8)
        
        video_writer.append_data(render_img_1008.cpu().numpy())

    cap.release()
    video_writer.close()
    
    t_pipeline_end = time.perf_counter()

    print("\n===================================")
    print(" [Ultra-Fast Streaming Pipeline Profiling]")
    print(f" - Total Frames Processed: {total_frames}")
    print(" - Total Processing Time : {:.2f} s".format(t_pipeline_end - t_pipeline_start))
    print(" - Average FPS           : {:.2f} FPS".format(total_frames / (t_pipeline_end - t_pipeline_start)))
    print("===================================\n")

if __name__ == "__main__":
    # 💡 파일 이름을 정확하게 지정해 주세요!
    VID_PATH = "input_video.mp4" 
    CKPT_PATH = "experiments/VDA_GPS_0529_Finetune/ckpt/VDA_GPS_0529_Finetune_final.pth"
    ENGINE_DIR = "trt_engines"
    REF_INTR_PATH = "../thuman_120cm_render_data_mono/mono_uniform_504/val/parm/0000_000/0_intr.npy"
    OUT_DIR = "inference_results"

    if not os.path.exists(VID_PATH):
        print(f"\n[Error] 동영상 파일을 찾을 수 없습니다: {VID_PATH}")
    else:
        run_ultimate_streaming_inference(VID_PATH, CKPT_PATH, ENGINE_DIR, REF_INTR_PATH, OUT_DIR)