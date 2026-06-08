import os
import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import math
import imageio
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import tensorrt as trt
from PIL import Image, ImageOps  # 💡 ImageOps 추가

from config.stereo_human_config import ConfigStereoHuman as config
from lib.network import VDAGaussianModel
from lib.GaussianRender import pts2render
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from lib.utils import depth2pc

import warnings
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")


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

# 💡 OpenCV를 배제하고 GPU 텐서상에서 바로 동작하는 패딩 함수 도입
def pad_to_square_tensor(tensor):
    h, w = tensor.shape[2], tensor.shape[3]
    max_side = max(h, w)
    pad_t = (max_side - h) // 2
    pad_b = max_side - h - pad_t
    pad_l = (max_side - w) // 2
    pad_r = max_side - w - pad_l
    return F.pad(tensor, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0.0)

def run_mixed_inference(img_path, ckpt_path, engine_dir, ref_intr_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_name = Path(img_path).stem

    cfg_wrapper = config()
    cfg_wrapper.load("config/stage2.yaml")
    cfg = cfg_wrapper.get_cfg()
    cfg.defrost()
    cfg.dataset.src_res = 504
    cfg.freeze()

    print("Loading PyTorch VDA Model Checkpoint...")
    model = VDAGaussianModel(cfg, with_gs_render=True)
    model.cuda()
    model.eval()
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['network'], strict=True)

    print("Loading TensorRT Engines...")
    unet_trt = TRTWrapper(os.path.join(engine_dir, "unet_extractor.engine"))
    gs_trt = TRTWrapper(os.path.join(engine_dir, "gs_regresser.engine"))

    print("Loading RVM Model...")
    hub_dir = torch.hub.get_dir()
    rvm_cache_path = os.path.join(hub_dir, "PeterL1n_RobustVideoMatting_master")

    if os.path.exists(rvm_cache_path):
        rvm = torch.hub.load(rvm_cache_path, "mobilenetv3", source='local').cuda().eval()
    else:
        rvm = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda().eval()

    # 💡 1. RVM을 포함한 모든 모델 Warm-up 선행 (지표 측정 분리)
    print("Running GPU Warm-up (RVM & Networks)...")
    t_warmup_start = time.perf_counter()
    dummy_img = torch.zeros((1, 3, 504, 504), device='cuda', dtype=torch.float32)
    with torch.no_grad():
        _ = rvm(dummy_img)
        _ = unet_trt(input_image_gps=dummy_img * 2.0 - 1.0)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            _, _ = model.vda_model(dummy_img.unsqueeze(1))
    torch.cuda.synchronize()
    t_warmup_end = time.perf_counter()
    time_warmup = (t_warmup_end - t_warmup_start) * 1000

    t_pipeline_start = time.perf_counter()

    # 💡 2. 100% GPU VRAM 전처리 로직 시작
    print("Processing Image (Pure GPU Pipeline)...")
    t_prep_start = time.perf_counter()
    
    # 💡 [문제 1 해결] EXIF 메타데이터를 확인하여 이미지 자동 회전 보정
    pil_img = Image.open(img_path).convert("RGB")
    pil_img = ImageOps.exif_transpose(pil_img) 
    
    img_tensor_raw = torch.from_numpy(np.array(pil_img)).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
    
    with torch.no_grad():
        # 💡 [문제 2 해결] 고해상도 이미지 RVM 최적화 (downsample_ratio 적용)
        # 입력 이미지의 해상도에 맞춰 RVM 내부 처리 비율을 자동으로 조절합니다.
        b, c, h, w = img_tensor_raw.shape
        ds_ratio = min(1.0, 512.0 / max(h, w)) 
        
        # fgr: 전경, pha: 알파 마스크
        fgr, pha, *rec = rvm(img_tensor_raw, downsample_ratio=ds_ratio)
    
    # 💡 [문제 2 해결] 마스크 임계값 완화 및 Soft Alpha Blending 적용
    # 검은색 옷이 배경으로 날아가는 것을 방지하기 위해 threshold를 낮춥니다.
    mask_tensor_raw = (pha > 0.15).float() 
    
    # 테두리가 픽셀 단위로 깨지는 것을 막기 위해 이진 마스크 대신 알파(pha)를 직접 곱해 부드럽게 합성합니다.
    img_clean_tensor = img_tensor_raw * pha 

    # GPU 텐서 레벨 패딩
    padded_img = pad_to_square_tensor(img_clean_tensor)
    padded_mask = pad_to_square_tensor(mask_tensor_raw)

    # GPU 텐서 레벨 리사이즈 (Interpolate)
    infer_res = 504
    img_tensor = F.interpolate(padded_img, size=(infer_res, infer_res), mode='area')
    mask_tensor = F.interpolate(padded_mask, size=(infer_res, infer_res), mode='nearest')

    raw_intr = np.load(ref_intr_path)
    ref_extr_path = ref_intr_path.replace('intr.npy', 'extr.npy')
    extr_np = np.load(ref_extr_path)
    if extr_np.shape == (3, 4):
        extr_4x4 = np.eye(4, dtype=np.float32)
        extr_4x4[:3, :] = extr_np
        extr_np = extr_4x4

    intr_504 = raw_intr.copy()  # 💡 이 줄을 추가해 주세요
    intr_tensor_504 = torch.from_numpy(raw_intr.copy()).float().unsqueeze(0).cuda()
    extr_tensor = torch.from_numpy(extr_np).float().unsqueeze(0).cuda()

    data = {'view_0': {'img': img_tensor, 'mask': mask_tensor, 'intr': intr_tensor_504, 'extr': extr_tensor}}
    torch.cuda.synchronize()
    t_prep_end = time.perf_counter()

    # 3. 네트워크 추론 (기존 최적화 로직 유지)
    print("Benchmarking Mixed Pipeline Sub-modules...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    img_gps = img_tensor * 2.0 - 1.0

    with torch.no_grad():
        start_event.record()
        unet_out = unet_trt(input_image_gps=img_gps)
        img_feat = (unet_out['feat1'], unet_out['feat2'], unet_out['feat3'])
        end_event.record()
        torch.cuda.synchronize()
        time_unet = start_event.elapsed_time(end_event)

        start_event.record()
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            depth_seq, _ = model.vda_model(img_tensor.unsqueeze(1))
        end_event.record()
        torch.cuda.synchronize()
        time_vda = start_event.elapsed_time(end_event)

        depth_pred = depth_seq.squeeze(1).unsqueeze(1).float()

        start_event.record()
        gs_out = gs_trt(img_gps=img_gps, depth=depth_pred, feat1=img_feat[0], feat2=img_feat[1], feat3=img_feat[2])
        rot = gs_out['rot_maps']
        scale = gs_out['scale_maps']
        opacity = gs_out['opacity_maps']
        end_event.record()
        torch.cuda.synchronize()
        time_gs = start_event.elapsed_time(end_event)

    bs = img_tensor.shape[0]
    data['view_0']['depth'] = depth_pred
    data['view_0']['xyz'] = depth2pc(depth_pred, extr_tensor, intr_tensor_504).view(bs, -1, 3)

    valid_mask = (depth_pred > 0.1).view(bs, -1)
    depth_flat = depth_pred.view(valid_mask.shape)
    mask_flat = mask_tensor.view(valid_mask.shape)

    data['view_0']['pts_valid'] = valid_mask & (depth_flat > 0.3) & (mask_flat > 0.5)
    data['view_0']['rot_maps'] = rot.view(bs, 4, infer_res, infer_res)
    data['view_0']['scale_maps'] = scale.view(bs, 3, infer_res, infer_res)
    data['view_0']['opacity_maps'] = opacity.view(bs, 1, infer_res, infer_res)

    render_res = 504
    fovx = focal2fov(intr_504[0, 0], render_res)
    fovy = focal2fov(intr_504[1, 1], render_res)
    proj_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, K=intr_504, h=render_res, w=render_res).transpose(0, 1).cuda()

    frames = []
    steps = np.linspace(0, 2 * np.pi, 60)
    bg_color = getattr(cfg.dataset, 'bg_color', [0.0, 0.0, 0.0])
    base_extr = extr_tensor[0]

    t_render_start = time.perf_counter()
    for step in tqdm(steps, desc="Rendering & Upsampling to 1008p"):
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
            'FovX': torch.tensor([fovx], dtype=torch.float32).cuda(),
            'FovY': torch.tensor([fovy], dtype=torch.float32).cuda(),
            'world_view_transform': world_view_transform.unsqueeze(0),
            'full_proj_transform': full_proj_transform.unsqueeze(0),
            'camera_center': cam_center.unsqueeze(0)
        }

        render_dict = {'lmain': data['view_0'], 'novel_view': data['novel_view']}

        with torch.no_grad():
            render_out = pts2render(render_dict, bg_color=bg_color, is_train=False)

        render_tensor = render_out['novel_view']['img_pred'][0]
        render_img_504 = render_tensor.detach().permute(1, 2, 0).cpu().numpy()

        target_res = 1008
        render_img_1008 = cv2.resize(render_img_504, (target_res, target_res), interpolation=cv2.INTER_LANCZOS4)
        frames.append(np.clip(render_img_1008 * 255.0, 0, 255).astype(np.uint8))
    t_render_end = time.perf_counter()

    t_io_start = time.perf_counter()
    out_mp4 = os.path.join(out_dir, out_name + "_Mixed_RVM_panning.mp4")
    imageio.mimsave(out_mp4, frames, fps=30, macro_block_size=1)

    sample_img = frames[15]
    out_img = os.path.join(out_dir, out_name + "_Mixed_RVM_sample.jpg")
    cv2.imwrite(out_img, cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR))
    t_io_end = time.perf_counter()
    
    t_pipeline_end = time.perf_counter()

    print("\n===================================")
    print(" [End-to-End Pipeline Profiling (Pure GPU RVM)]")
    print(" - Pre-processing (RVM)  : {:.2f} ms".format((t_prep_end - t_prep_start) * 1000))
    print(" - Warm-up (Cold Start)  : {:.2f} ms".format(time_warmup))
    print(" - Network Inference     : {:.2f} ms".format(time_unet + time_vda + time_gs))
    print("   ├─ U-Net (TRT)        : {:.2f} ms".format(time_unet))
    print("   ├─ VDA (PT BF16)      : {:.2f} ms".format(time_vda))
    print("   └─ GS Regresser (TRT) : {:.2f} ms".format(time_gs))
    print(" - Total CUDA Operation  : {:.2f} ms".format(time_warmup + time_unet + time_vda + time_gs))
    print(" - Render & Upsample (60): {:.2f} ms".format((t_render_end - t_render_start) * 1000))
    print(" - Video & Image I/O     : {:.2f} ms".format((t_io_end - t_io_start) * 1000))
    print("-----------------------------------")
    print(" - Total Pipeline Time   : {:.2f} ms".format((t_pipeline_end - t_pipeline_start) * 1000))
    print("===================================\n")

if __name__ == "__main__":
    IMG_PATH = "test.jpg"
    CKPT_PATH = "experiments/VDA_GPS_0529_Finetune/ckpt/VDA_GPS_0529_Finetune_final.pth"
    ENGINE_DIR = "trt_engines"
    REF_INTR_PATH = "../thuman_120cm_render_data_mono/mono_uniform_504/val/parm/0000_000/0_intr.npy"
    OUT_DIR = "inference_results"

    run_mixed_inference(IMG_PATH, CKPT_PATH, ENGINE_DIR, REF_INTR_PATH, OUT_DIR)