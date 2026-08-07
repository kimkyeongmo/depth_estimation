# Copyright (2025) Bytedance Ltd. and/or its affiliates
# ... (라이선스 생략) ...

import argparse
import numpy as np
import os
import torch
import time

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything Inference')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    # 💡 [추가] 사용자가 원하는 가중치 파일 경로를 직접 입력할 수 있도록 추가
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the fine-tuned checkpoint (.pth)')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl']) # vits를 기본값으로 변경
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--metric', action='store_true', help='use metric model')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr')
    parser.add_argument('--focal-length-x', default=470.4, type=float, help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=470.4, type=float, help='Focal length along the y-axis.')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    print(f"Loading {args.encoder.upper()} model...")
    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder], metric=args.metric)
    
    # --- [수정] 가중치 로드 부분: 사용자가 입력한 경로 사용 및 compile 접두사 제거 ---
    print(f"[Info] Loading weights from: {args.ckpt_path}")
    state_dict = torch.load(args.ckpt_path, map_location='cpu')
    
    # torch.compile 로 학습된 가중치의 '_orig_mod.' 접두사 제거 로직
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('_orig_mod.', '')
        clean_state_dict[clean_key] = v
        
    video_depth_anything.load_state_dict(clean_state_dict, strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    # --- [1. 로드 단계] ---
    load_start = time.time()
    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    load_duration = time.time() - load_start
    
    num_frames = len(frames)
    print(f"Start inference for {num_frames} frames...")
    
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats() 
        torch.cuda.synchronize()
    
    # --- [2. 순수 추론 단계] ---
    inf_start_time = time.time()

    with torch.no_grad(): # 추론 모드에서는 gradient 계산 제외
        depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

    if DEVICE == 'cuda': torch.cuda.synchronize()
    inf_duration = time.time() - inf_start_time

    # --- [3. 동영상 저장 단계 (PLY 제외)] ---
    print("Inference complete. Starting video & file I/O...")
    save_start = time.time()
    video_name = os.path.basename(args.input_video)
    os.makedirs(args.output_dir, exist_ok=True)

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
    
    # 원본 프레임 저장
    save_video(frames, processed_video_path, fps=fps)
    
    # 깜빡임 방지 및 Spectral(V2) 컬러맵 직접 적용
    import matplotlib
    import cv2
    cmap = matplotlib.colormaps.get_cmap('Spectral') 
    
    # 비디오 전체를 아우르는 절대 기준점 (Global Min/Max) 계산
    global_min = np.min(depths)
    global_max = np.max(depths)
    
    colored_depths = []
    for depth in depths:
        # 매 프레임 고정된 글로벌 스케일로 정규화
        if global_max - global_min > 0:
            depth_norm = (depth - global_min) / (global_max - global_min)
        else:
            depth_norm = depth
            
        depth_color = (cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        colored_depths.append(depth_color)

    # shape 에러를 막기 위해 리스트를 Numpy 배열로 변환
    colored_depths = np.array(colored_depths) 

    # 색칠한 배열을 넘기고, is_depths=False로 설정하여 그대로 저장하게 만듦
    save_video(colored_depths, depth_vis_path, fps=fps, is_depths=False)

    if args.save_npz:
        depth_npz_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths.npz')
        np.savez_compressed(depth_npz_path, depths=depths)
    if args.save_exr:
        depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
        os.makedirs(depth_exr_dir, exist_ok=True)
        import OpenEXR
        import Imath
        for i, depth in enumerate(depths):
            output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
            header = OpenEXR.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()

    save_duration = time.time() - save_start
    
    # 파이프라인 전체 시간 = 로드 + 추론 + 영상저장 (PLY 제외)
    pipeline_total_duration = load_duration + inf_duration + save_duration
    
    # --- [성능 지표 계산 및 출력] ---
    avg_inf_latency = (inf_duration / num_frames) * 1000
    avg_inf_fps = num_frames / inf_duration

    avg_pipe_latency = (pipeline_total_duration / num_frames) * 1000
    avg_pipe_fps = num_frames / pipeline_total_duration

    print("\n" + "="*55)
    print(f" VIDEO-DEPTH-ANYTHING PERFORMANCE SUMMARY")
    print(f" Encoder             : {args.encoder.upper()}")
    print(f" Total Frames        : {num_frames}")
    print(f"-"*55)
    print(f" [1. Pure Inference (Model Only)]")
    print(f"  - Avg Latency : {avg_inf_latency:.2f} ms")
    print(f"  - Avg FPS     : {avg_inf_fps:.2f} FPS")
    print(f"\n [2. Total Pipeline (Load + Infer + Video Save)]")
    print(f"  - Avg Latency : {avg_pipe_latency:.2f} ms")
    print(f"  - Avg FPS     : {avg_pipe_fps:.2f} FPS")
    print(f"  - Load Time   : {load_duration:.2f} sec")
    print(f"  - Save Time   : {save_duration:.2f} sec")
    print(f"  - Total Time  : {pipeline_total_duration:.2f} sec")
    
    if DEVICE == 'cuda':
        print(f"-"*55)
        max_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f" Peak VRAM     : {max_vram:.2f} MB")
    print("="*55 + "\n")

    # --- [4. PLY 파일 생성 (시간 측정에서 완전히 제외됨)] ---
    if args.metric:
        print("Generating 3D Point Clouds (.ply) ... (This is NOT included in FPS calculation)")
        import open3d as o3d
        
        ply_out_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0] + '_ply')
        os.makedirs(ply_out_dir, exist_ok=True)

        width, height = depths[0].shape[-1], depths[0].shape[-2]
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / args.focal_length_x
        y = (y - height / 2) / args.focal_length_y
        
        for i, (color_image, depth) in enumerate(zip(frames, depths)):
            #z = np.array(depth)
            scale_factor = 1.8 # 찌그러진 정도에 따라 조절
            z = np.array(depth) * scale_factor
            points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            colors = np.array(color_image).reshape(-1, 3) / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(os.path.join(ply_out_dir, 'point_' + str(i).zfill(4) + '.ply'), pcd)
            
            if (i + 1) % 10 == 0 or (i + 1) == num_frames:
                print(f"  -> Saved point cloud {i + 1} / {num_frames}")
                
        print(f"\nAll PLY files saved successfully to: {ply_out_dir}")