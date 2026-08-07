import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.add_dll_directory("C:/Users/COM/miniconda3/envs/gps_gaussian/lib/site-packages/torch/lib")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin")
#from pybind11 import CudaRuntime1

import copy
import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import imageio
import threading 
from pathlib import Path
from tqdm import tqdm
import tensorrt as trt
import warnings
import pyglet
from pyglet.gl import *

warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

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


class LatestFrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.stopped = False

    def update(self, frame):
        with self.lock:
            self.frame = frame
            self.frame_id += 1

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, self.frame_id
            return self.frame.copy(), self.frame_id

    def stop(self):
        with self.lock:
            self.stopped = True

    def is_stopped(self):
        with self.lock:
            return self.stopped


def capture_latest_loop(stream_url, latest_buffer):
    """별도 thread에서 스트림 받아 LatestFrameBuffer에 최신 프레임 저장."""
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 내부 버퍼 최소화

    if not cap.isOpened():
        latest_buffer.stop()
        print(f"[STREAM] Cannot open: {stream_url}")
        return

    print(f"[STREAM] opened: {stream_url}")
    print(f"[STREAM] reported fps = {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"[STREAM] reported resolution = "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    read_count = 0
    fail_count = 0
    while not latest_buffer.is_stopped():
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count > 100:
                print("[STREAM] too many read failures, stopping")
                break
            time.sleep(0.01)
            continue
        fail_count = 0
        latest_buffer.update(frame)
        read_count += 1
        if read_count % 100 == 0:
            print(f"[STREAM] read_count={read_count}")

    cap.release()
    latest_buffer.stop()
    print("[STREAM] capture thread stopped")
# ============================================================


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


def run_ultimate_streaming_inference_filtered(
    vid_path, ckpt_path, engine_dir, ref_intr_path, out_dir,
    is_stream=False,                # 🆕 SRT/RTSP 입력 여부
    max_process_frames=-1,          # 🆕 무한실행 시 -1, 테스트 시 양수
):
    os.makedirs(out_dir, exist_ok=True)
    out_name = Path(vid_path).stem if not is_stream else "stream_output"
    #CudaRuntime1.init_window(3840, 1080)

    cfg_wrapper = config()
    cfg_wrapper.load("config/stage2.yaml")
    cfg = cfg_wrapper.get_cfg()
    cfg.defrost()
    cfg.dataset.src_res = 504
    cfg.freeze()

    base_model = VDAGaussianModel(cfg, with_gs_render=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    base_model.load_state_dict(ckpt['network'], strict=True)
    vda_weights = base_model.vda_model.state_dict()
    del base_model

    vda_stream = StreamingVDA(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    vda_stream.load_state_dict(vda_weights, strict=True)
    vda_stream = vda_stream.cuda().eval()

    unet_trt = TRTWrapper(os.path.join(engine_dir, "unet_extractor.engine"))
    gs_trt = TRTWrapper(os.path.join(engine_dir, "gs_regresser.engine"))

    hub_dir = torch.hub.get_dir()
    rvm_cache_path = os.path.join(hub_dir, "PeterL1n_RobustVideoMatting_master")
    if os.path.exists(rvm_cache_path):
        rvm = torch.hub.load(rvm_cache_path, "mobilenetv3", source='local').cuda().eval()
    else:
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
    #target_res = 1008
    target_w = 1920
    target_h = 1080
    fovx = focal2fov(intr_504[0, 0], render_res)
    fovy = focal2fov(intr_504[1, 1], render_res)
    proj_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, K=intr_504, h=render_res, w=render_res).transpose(0, 1).cuda()
    bg_color = getattr(cfg.dataset, 'bg_color', [0.0, 0.0, 0.0])
    base_extr = extr_tensor[0]

    # ============================================================
    # 🆕 입력 소스: 스트림 vs 파일 분기
    # ============================================================
    if is_stream:
        latest_buffer = LatestFrameBuffer()
        capture_thread = threading.Thread(
            target=capture_latest_loop,
            args=(vid_path, latest_buffer),
            daemon=True,
        )
        capture_thread.start()

        print("[STREAM] waiting for first frame...")
        while True:
            first_frame, _ = latest_buffer.read()
            if first_frame is not None:
                break
            if latest_buffer.is_stopped():
                raise RuntimeError("Capture thread stopped before receiving frames.")
            time.sleep(0.01)

        total_frames = -1
        fps = 30.0
        vid_rotation = 0  # 스트림에선 보통 메타데이터 없음, 필요시 따로 처리
    else:
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps): fps = 30.0
        vid_rotation = get_video_rotation(vid_path)
        latest_buffer = None
        capture_thread = None
    # ============================================================

    out_mp4 = os.path.join(out_dir, out_name + "_Filtered_Stream.mp4")
    video_writer = imageio.get_writer(out_mp4, fps=fps, macro_block_size=1)

    rec = [None] * 4
    prev_depth, prev_rot, prev_scale, prev_opacity = None, None, None, None
    outlier_threshold = 0.01
    ema_alpha = 0.9

    time_vram_total = 0.0
    time_prep = 0.0
    time_unet = 0.0
    time_vda = 0.0
    time_gs = 0.0
    time_render = 0.0
    time_io = 0.0
    processed_frames = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    t_pipeline_start = time.perf_counter()

    # 🆕 스트림 모드용 변수
    last_processed_frame_id = -1
    skipped_frames = 0
    frame_idx = 0

    # 🆕 진행률 표시: 파일이면 total_frames, 스트림이면 max_process_frames 또는 None
    if is_stream:
        pbar_total = max_process_frames if max_process_frames > 0 else None
        pbar = tqdm(total=pbar_total, desc="SRT Stream Inference")
    else:
        pbar = tqdm(total=total_frames, desc="Robust Streaming Inference")

    # ============================================================
    # 🆕 메인 loop: 스트림 vs 파일 양쪽 지원
    # ============================================================
    while True:
        # --- 종료 조건 ---
        if is_stream:
            if latest_buffer.is_stopped():
                print("[STREAM] capture stopped, exiting main loop")
                break
            if max_process_frames > 0 and processed_frames >= max_process_frames:
                print(f"[STREAM] max_process_frames reached: {max_process_frames}")
                break
        else:
            if frame_idx >= total_frames:
                break

        # --- 프레임 획득 ---
        t_io_start = time.perf_counter()

        if is_stream:
            frame, current_frame_id = latest_buffer.read()
            if frame is None:
                time.sleep(0.005)
                continue
            # 같은 프레임 중복 처리 방지
            if current_frame_id == last_processed_frame_id:
                time.sleep(0.001)
                continue
            # 처리 못한 프레임 카운트
            if last_processed_frame_id >= 0:
                skipped_frames += max(0, current_frame_id - last_processed_frame_id - 1)
            last_processed_frame_id = current_frame_id
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # --- 회전 ---
        if vid_rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif vid_rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif vid_rotation == 270 or vid_rotation == -90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t_io_end = time.perf_counter()
        time_io += (t_io_end - t_io_start) * 1000

        # 💡 VRAM Allocation 격리
        t_vram_start = time.perf_counter()
        img_tensor_raw = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
        torch.cuda.synchronize()
        t_vram_end = time.perf_counter()
        time_vram_total += (t_vram_end - t_vram_start)

        t_prep_start = time.perf_counter()
        with torch.no_grad():
            b, c, h, w = img_tensor_raw.shape
            ds_ratio = min(1.0, 512.0 / max(h, w))
            fgr, pha, *rec = rvm(img_tensor_raw, *rec, downsample_ratio=ds_ratio)
            img_clean_tensor = img_tensor_raw * pha
            padded_img = pad_to_square_tensor(img_clean_tensor)
            padded_mask = pad_to_square_tensor(pha)
            img_tensor = F.interpolate(padded_img, size=(render_res, render_res), mode='area')
            mask_tensor = F.interpolate(padded_mask, size=(render_res, render_res), mode='nearest')
            img_504_np = (img_tensor[0].permute(1, 2, 0).detach() * 255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)
        torch.cuda.synchronize()
        t_prep_end = time.perf_counter()
        time_prep += (t_prep_end - t_prep_start) * 1000

        start_event.record()
        depth_np = vda_stream.infer_video_depth_one(img_504_np, input_size=504, device='cuda', fp32=True)
        end_event.record()
        torch.cuda.synchronize()
        time_vda += start_event.elapsed_time(end_event)

        depth_pred = torch.from_numpy(depth_np).float().unsqueeze(0).unsqueeze(0).cuda()
        img_gps = img_tensor * 2.0 - 1.0

        with torch.no_grad():
            start_event.record()
            unet_out = unet_trt(input_image_gps=img_gps)
            img_feat = (unet_out['feat1'], unet_out['feat2'], unet_out['feat3'])
            end_event.record()
            torch.cuda.synchronize()
            time_unet += start_event.elapsed_time(end_event)

            start_event.record()
            gs_out = gs_trt(img_gps=img_gps, depth=depth_pred, feat1=img_feat[0], feat2=img_feat[1], feat3=img_feat[2])
            rot = gs_out['rot_maps']
            scale = gs_out['scale_maps']
            opacity = gs_out['opacity_maps']

            if prev_depth is None:
                prev_depth, prev_rot, prev_scale, prev_opacity = depth_pred, rot, scale, opacity
            else:
                global_diff = torch.abs(depth_pred - prev_depth).mean().item()
                if global_diff > outlier_threshold:
                    depth_pred = prev_depth
                    rot = prev_rot
                    scale = prev_scale
                    opacity = prev_opacity
                else:
                    depth_pred = ema_alpha * depth_pred + (1.0 - ema_alpha) * prev_depth
                    rot = F.normalize(ema_alpha * rot + (1.0 - ema_alpha) * prev_rot, p=2, dim=1, eps=1e-6)
                    scale = ema_alpha * scale + (1.0 - ema_alpha) * prev_scale
                    opacity = ema_alpha * opacity + (1.0 - ema_alpha) * prev_opacity
                    prev_depth, prev_rot, prev_scale, prev_opacity = depth_pred, rot, scale, opacity

            end_event.record()
            torch.cuda.synchronize()
            time_gs += start_event.elapsed_time(end_event)

        t_render_start = time.perf_counter()
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

        test_ipd = 0.6
        proj_mat = proj_matrix

        data_left = copy.deepcopy({'view_0': data['view_0'], 'novel_view': data['novel_view']})
        data_left['novel_view']['world_view_transform'][0, 3, 0] -= (test_ipd / 2.0)
        data_left['novel_view']['full_proj_transform'] = torch.bmm(data_left['novel_view']['world_view_transform'], proj_mat.unsqueeze(0))
        data_left['novel_view']['camera_center'] = data_left['novel_view']['world_view_transform'].inverse()[0, 3, :3].unsqueeze(0)

        data_right = copy.deepcopy({'view_0': data['view_0'], 'novel_view': data['novel_view']})
        data_right['novel_view']['world_view_transform'][0, 3, 0] += (test_ipd / 2.0)
        data_right['novel_view']['full_proj_transform'] = torch.bmm(data_right['novel_view']['world_view_transform'], proj_mat.unsqueeze(0))
        data_right['novel_view']['camera_center'] = data_right['novel_view']['world_view_transform'].inverse()[0, 3, :3].unsqueeze(0)

        with torch.no_grad():
            render_left, render_right = pts2render(
                {'lmain': data_left['view_0'], 'novel_view': data_left['novel_view']},
                {'lmain': data_right['view_0'], 'novel_view': data_right['novel_view']},
                bg_color=bg_color, is_train=False
            )

        render_tensor_left_504 = render_left['novel_view']['img_pred']
        render_tensor_right_504 = render_right['novel_view']['img_pred']
        render_tensor_left_1008 = F.interpolate(render_tensor_left_504, size=(target_h, target_w), mode='bicubic', align_corners=False)
    
        render_tensor_right_1008 = F.interpolate(render_tensor_right_504, size=(target_h, target_w), mode='bicubic', align_corners=False)

        shift = 300
        render_tensor_left_1008 = render_tensor_left_1008[..., :-shift]
        render_tensor_right_1008 = render_tensor_right_1008[..., shift:]
        render_tensor_left_1008 = F.pad(render_tensor_left_1008, (shift, 0), mode='constant', value=0.0)
        render_tensor_right_1008 = F.pad(render_tensor_right_1008, (0, shift), mode='constant', value=0.0)
        render_tensor_1008 = torch.cat([render_tensor_left_1008, render_tensor_right_1008], dim=3)
        render_chw = render_tensor_1008[0]
        render_hwc = render_chw.permute(1, 2, 0)
        alpha = torch.ones(render_hwc.shape[0], render_hwc.shape[1], 1, device=render_hwc.device)
        render_rgba = torch.cat([render_hwc, alpha], dim=2).contiguous()
        #CudaRuntime1.show_tensor(render_rgba)
        render_img_1008 = (render_tensor_1008[0].detach().permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8)
        torch.cuda.synchronize()

        t_render_end = time.perf_counter()
        time_render += (t_render_end - t_render_start) * 1000

        t_io_start = time.perf_counter()
        video_writer.append_data(render_img_1008.cpu().numpy())
        t_io_end = time.perf_counter()
        torch.cuda.synchronize()
        time_io += (t_io_end - t_io_start) * 1000

        processed_frames += 1
        frame_idx += 1
        pbar.update(1)

        if is_stream and processed_frames % 30 == 0:
            elapsed_now = time.perf_counter() - t_pipeline_start
            print(f"[STREAM] processed={processed_frames}, "
                  f"latest_id={current_frame_id}, skipped={skipped_frames}, "
                  f"fps={processed_frames / elapsed_now:.2f}")

    if is_stream:
        latest_buffer.stop()
        capture_thread.join(timeout=2.0)
    else:
        cap.release()
    pbar.close()
    video_writer.close()
    #CudaRuntime1.cleanup()
    t_pipeline_end = time.perf_counter()

    avg_prep = time_prep / processed_frames if processed_frames > 0 else 0
    avg_unet = time_unet / processed_frames if processed_frames > 0 else 0
    avg_vda = time_vda / processed_frames if processed_frames > 0 else 0
    avg_gs = time_gs / processed_frames if processed_frames > 0 else 0
    avg_render = time_render / processed_frames if processed_frames > 0 else 0
    avg_io = time_io / processed_frames if processed_frames > 0 else 0

    avg_network = avg_unet + avg_vda + avg_gs
    avg_cuda_pure = avg_prep + avg_network + avg_render

    print("\n===================================")
    print(" [End-to-End Pipeline Profiling (Filtered Streaming)]")
    print(" - VRAM Allocation & Data Transfer : {:.2f} s (Total Time)".format(time_vram_total))
    print("-----------------------------------")
    print(" [Per-Frame GPU Inference Time]")
    print(" - Pre-processing (RVM)  : {:.2f} ms".format(avg_prep))
    print(" - Network Inference     : {:.2f} ms".format(avg_network))
    print("   ├─ U-Net (TRT)        : {:.2f} ms".format(avg_unet))
    print("   ├─ VDA Stream (PT)    : {:.2f} ms".format(avg_vda))
    print("   └─ GS Regresser (TRT) : {:.2f} ms".format(avg_gs))
    print(" - Render & Upsample     : {:.2f} ms".format(avg_render))
    print("   = Pure CUDA Operation : {:.2f} ms / frame".format(avg_cuda_pure))
    print("-----------------------------------")
    print(" - Video & Image I/O     : {:.2f} ms / frame".format(avg_io))
    print(" - Total Pipeline Time   : {:.2f} s".format(t_pipeline_end - t_pipeline_start))
    if is_stream:
        print(" - Skipped Input Frames  : {}".format(skipped_frames))
    if t_pipeline_end - t_pipeline_start > 0:
        print(" - Average Target FPS    : {:.2f} FPS".format(processed_frames / (t_pipeline_end - t_pipeline_start)))
    print("===================================\n")


if __name__ == "__main__":
    
    USE_STREAM = True 

    if USE_STREAM:
        
        VID_PATH = "rtsp://localhost:8554/mystream"
        IS_STREAM = True
        MAX_FRAMES = 300
    else:
        VID_PATH = "input_video.mp4"
        IS_STREAM = False
        MAX_FRAMES = -1

    CKPT_PATH = "checkpoints/VDA_GPS_0529_Finetune_final.pth"
    ENGINE_DIR = "trt_engines"
    REF_INTR_PATH = "val/parm/0000_000/0_intr.npy"
    OUT_DIR = "inference_results"

    if not IS_STREAM and not os.path.exists(VID_PATH):
        print(f"\n[Error] 비디오 파일을 찾을 수 없습니다: {VID_PATH}")
    else:
        run_ultimate_streaming_inference_filtered(
            VID_PATH, CKPT_PATH, ENGINE_DIR, REF_INTR_PATH, OUT_DIR,
            is_stream=IS_STREAM,
            max_process_frames=MAX_FRAMES,
        )