"""
THuman 2.0 (GPS-Gaussian val 포맷) 단일 이미지 평가 스크립트

  - 입력: val/img/{subject}/{src_view}.jpg
  - 목표: val/img/{subject}/{tgt_view}.jpg   (GT)
  - novel view 카메라를 GT 뷰의 extrinsic 으로 설정해 PSNR / SSIM / LPIPS 계산
  - 지표는 인물 전경 바운딩 박스에서만 계산 (검은 배경으로 인한 수치 과대평가 방지)
"""
import os
import sys
import glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.add_dll_directory("C:/Users/COM/miniconda3/envs/gps_gaussian/lib/site-packages/torch/lib")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin")

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from config.stereo_human_config import ConfigStereoHuman as config
from lib.GaussianRender import pts2render
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from lib.utils import depth2pc

from inference_trt_video_filtered import TRTWrapper, pad_to_square_tensor

cv2.setNumThreads(0)
torch.set_float32_matmul_precision('high')

_LPIPS_NET = None      # 스윕 시 매번 재로드하지 않도록 캐시


# --------------------------------------------------------------------------
# 카메라 유틸
# --------------------------------------------------------------------------
def load_cam(parm_dir, view_id):
    intr = np.load(os.path.join(parm_dir, f"{view_id}_intr.npy")).astype(np.float32)
    extr = np.load(os.path.join(parm_dir, f"{view_id}_extr.npy")).astype(np.float32)
    if extr.shape == (3, 4):
        e = np.eye(4, dtype=np.float32)
        e[:3, :] = extr
        extr = e
    return intr, extr


def view_angle(extr_a, extr_b):
    """두 카메라의 회전(orientation) 차이 [deg]"""
    Ra, Rb = extr_a[:3, :3], extr_b[:3, :3]
    R = Ra @ Rb.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def orbit_angle(extr_a, extr_b, center=np.zeros(3)):
    """피사체 중심에서 본 두 카메라 사이의 공전각 [deg]"""
    def cam_center(E):
        R, t = E[:3, :3], E[:3, 3]
        return -R.T @ t
    va = cam_center(extr_a) - center
    vb = cam_center(extr_b) - center
    cos = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def list_views(parm_dir):
    return sorted(int(os.path.basename(p).split('_')[0])
                  for p in glob.glob(os.path.join(parm_dir, "*_intr.npy")))


def build_novel_view(intr, extr, render_res):
    fovx = focal2fov(intr[0, 0], render_res)
    fovy = focal2fov(intr[1, 1], render_res)

    proj = getProjectionMatrix(
        znear=0.01, zfar=100.0, K=intr, h=render_res, w=render_res
    ).transpose(0, 1).cuda()

    extr_t = torch.from_numpy(extr).float().cuda()
    R = extr_t[:3, :3].T
    T = extr_t[:3, 3]

    wvt = torch.tensor(
        getWorld2View2(R.cpu().numpy(), T.cpu().numpy(),
                       np.array([0.0, 0.0, 0.0]), 1.0)
    ).transpose(0, 1).float().cuda()

    full_proj = wvt.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    cam_center = wvt.inverse()[3, :3]

    return {
        'width':  torch.tensor([render_res], dtype=torch.int32).cuda(),
        'height': torch.tensor([render_res], dtype=torch.int32).cuda(),
        'FovX':   torch.tensor([fovx], dtype=torch.float32).cuda(),
        'FovY':   torch.tensor([fovy], dtype=torch.float32).cuda(),
        'world_view_transform': wvt.unsqueeze(0),
        'full_proj_transform':  full_proj.unsqueeze(0),
        'camera_center':        cam_center.unsqueeze(0),
    }


# --------------------------------------------------------------------------
# 평가 지표
# --------------------------------------------------------------------------
def compute_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    return float('inf') if mse == 0 else 10.0 * np.log10(1.0 / mse)


def compute_ssim(pred, gt):
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(gt, pred, channel_axis=2, data_range=1.0)
    except ImportError:
        return None


def add_labels(img, labels=("Input", "Ours", "GT"), bar_ratio=0.11):
    """이미지 아래에 검은 띠를 덧붙이고 각 패널 중앙에 흰 라벨을 넣는다."""
    h, w = img.shape[:2]
    n, pw = len(labels), img.shape[1] // len(labels)
    bar_h = max(int(h * bar_ratio), 24)
    out = np.concatenate([img, np.zeros((bar_h, w, 3), np.uint8)], axis=0)

    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = bar_h / 38.0
    thick = max(int(round(scale * 1.4)), 1)
    for i, t in enumerate(labels):
        (tw, th), _ = cv2.getTextSize(t, font, scale, thick)
        x = i * pw + (pw - tw) // 2
        y = h + (bar_h + th) // 2
        cv2.putText(out, t, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def compute_lpips(pred, gt):
    global _LPIPS_NET
    try:
        import lpips
        if _LPIPS_NET is None:
            _LPIPS_NET = lpips.LPIPS(net='alex').cuda()
        p = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
        g = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).cuda() * 2 - 1
        with torch.no_grad():
            return _LPIPS_NET(p, g).item()
    except ImportError:
        return None


# --------------------------------------------------------------------------
# 메인 평가 루틴
# --------------------------------------------------------------------------
def eval_thuman(
    data_root="val",
    subject="0000_000",
    src_view=0,
    tgt_view=1,
    engine_dir="trt_engines",
    out_dir="eval_results",
    warmup=32,
    crop_margin=10,
    engines=None,          # 스윕 시 엔진 재사용
    verbose=True,
    depth_scale = 1.0
):
    os.makedirs(out_dir, exist_ok=True)
    render_res = 504

    cfg_wrapper = config()
    cfg_wrapper.load("config/stage2.yaml")
    cfg = cfg_wrapper.get_cfg()
    cfg.defrost(); cfg.dataset.src_res = render_res; cfg.freeze()
    bg_color = getattr(cfg.dataset, 'bg_color', [0.0, 0.0, 0.0])

    # ---------------- 엔진 ----------------
    if engines is None:
        if verbose:
            print("[INFO] loading TensorRT engines...")
        engines = {
            'vda':  TRTWrapper(os.path.join(engine_dir, "vda_model.engine")),
            'unet': TRTWrapper(os.path.join(engine_dir, "unet_extractor.engine")),
            'gs':   TRTWrapper(os.path.join(engine_dir, "gs_regresser.engine")),
        }
    vda_trt, unet_trt, gs_trt = engines['vda'], engines['unet'], engines['gs']

    # ---------------- 데이터 ----------------
    img_dir  = os.path.join(data_root, "img",  subject)
    parm_dir = os.path.join(data_root, "parm", subject)

    src_bgr = cv2.imread(os.path.join(img_dir, f"{src_view}.jpg"))
    gt_bgr  = cv2.imread(os.path.join(img_dir, f"{tgt_view}.jpg"))
    assert src_bgr is not None, f"입력 이미지 없음: {src_view}.jpg"
    assert gt_bgr  is not None, f"GT 이미지 없음: {tgt_view}.jpg"

    src_intr, src_extr = load_cam(parm_dir, src_view)
    tgt_intr, tgt_extr = load_cam(parm_dir, tgt_view)

    ang_rot   = view_angle(src_extr, tgt_extr)
    ang_orbit = orbit_angle(src_extr, tgt_extr)

    intr_tensor = torch.from_numpy(src_intr).float().unsqueeze(0).cuda()
    extr_tensor = torch.from_numpy(src_extr).float().unsqueeze(0).cuda()

    novel_view = build_novel_view(tgt_intr, tgt_extr, render_res)

    # ---------------- 마스크 (루프 밖에서 1회) ----------------
    src_raw = torch.from_numpy(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)) \
                   .float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0

    mask_path = os.path.join(data_root, "mask", subject, f"{src_view}.png")
    if os.path.exists(mask_path):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (src_bgr.shape[1], src_bgr.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
        pha = torch.from_numpy(m).float().cuda().view(1, 1, *m.shape) / 255.0
    else:
        pha = (src_raw.sum(dim=1, keepdim=True) > 0.02).float()

    img_clean = src_raw * pha

    h, w = img_clean.shape[2], img_clean.shape[3]
    scale_f   = render_res / max(h, w)
    new_short = int(round(min(h, w) * scale_f))
    new_size  = (new_short, render_res) if w >= h else (render_res, new_short)

    img_resized  = F.interpolate(img_clean, size=new_size, mode='bicubic',
                                 align_corners=False, antialias=True)
    mask_resized = F.interpolate(pha, size=new_size, mode='nearest')
    img_tensor   = pad_to_square_tensor(img_resized)
    mask_tensor  = pad_to_square_tensor(mask_resized)

    vda_mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    vda_std  = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    vda_ready = (img_tensor - vda_mean) / vda_std
    vda_input = vda_ready.unsqueeze(1).contiguous()

    # ---------------- VDA 시간적 캐시 워밍업 ----------------
    INFER_LEN, OVERLAP, INTERP_LEN = 32, 10, 8
    gap = (INFER_LEN - OVERLAP) * 2 - 1 - (OVERLAP - INTERP_LEN)
    frame_cache_list, frame_id_list = [], []
    vda_id, num_caches = -1, 0
    depth_pred = None

    with torch.no_grad():
        for _ in range(warmup):
            vda_id += 1
            inputs = {'input_image': vda_input}

            if vda_id == 0:
                num_caches = len([k for k in vda_trt.outputs if 'out_cache' in k])
                cur_cache = [
                    torch.zeros(tuple(vda_trt.engine.get_tensor_shape(f'in_cache_{i}')),
                                dtype=torch.float32, device='cuda')
                    for i in range(num_caches)
                ]
                for i, c in enumerate(cur_cache):
                    inputs[f'in_cache_{i}'] = c
                vda_out = vda_trt(**inputs)
                new_cache = [vda_out[f'out_cache_{i}'].clone() for i in range(num_caches)]
                frame_cache_list = [new_cache] * INFER_LEN
                frame_id_list.extend([0] * (INFER_LEN - 1))
            else:
                cur_list = frame_cache_list[0:2] + frame_cache_list[-INFER_LEN + 3:]
                cur_cache = [torch.cat([hh[i] for hh in cur_list], dim=1)
                             for i in range(len(cur_list[0]))]
                for i, c in enumerate(cur_cache):
                    inputs[f'in_cache_{i}'] = c
                vda_out = vda_trt(**inputs)
                new_cache = [vda_out[f'out_cache_{i}'].clone() for i in range(num_caches)]
                frame_cache_list.append(new_cache)
                frame_id_list.append(vda_id)
                if vda_id + INFER_LEN > gap + 1:
                    del frame_id_list[1]; del frame_cache_list[1]

            depth_pred = vda_out['depth_out'].view(1, 1, render_res, render_res)

        # ---------------- U-Net + Gaussian + Render ----------------
        img_gps  = img_tensor * 2.0 - 1.0
        unet_out = unet_trt(input_image_gps=img_gps)
        feats = (unet_out['feat1'], unet_out['feat2'], unet_out['feat3'])
        gs_out = gs_trt(img_gps=img_gps, depth=depth_pred,
                        feat1=feats[0], feat2=feats[1], feat3=feats[2])

        bs = img_tensor.shape[0]
        d = depth_pred.view(-1).cpu().numpy()
        m = mask_tensor.view(-1).cpu().numpy() > 0.5
        print("fg depth: min=%.4f max=%.4f mean=%.4f" % (d[m].min(), d[m].max(), d[m].mean()))
        print("cam dist :", np.linalg.norm(-src_extr[:3,:3].T @ src_extr[:3,3]))
        depth_for_pc = depth_pred * depth_scale
        view0 = {
            'img': img_tensor, 'mask': mask_tensor,
            'intr': intr_tensor, 'extr': extr_tensor,
            'depth': depth_for_pc,
            'xyz': depth2pc(depth_for_pc, extr_tensor, intr_tensor).view(bs, -1, 3),
            'rot_maps':     gs_out['rot_maps'].view(bs, 4, render_res, render_res),
            'scale_maps':   gs_out['scale_maps'].view(bs, 3, render_res, render_res),
            'opacity_maps': gs_out['opacity_maps'].view(bs, 1, render_res, render_res),
            
        }
        valid = (depth_pred > 0.05).view(bs, -1)
        view0['pts_valid'] = valid & (mask_tensor.view(valid.shape) > 0.5)

        rendered, _ = pts2render(
            {'lmain': view0, 'novel_view': novel_view},
            {'lmain': view0, 'novel_view': novel_view},
            bg_color=bg_color, is_train=False
        )
        render_np = rendered['novel_view']['img_pred'][0] \
                        .permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()

    # ---------------- GT 정렬 ----------------
    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gt_rgb = cv2.resize(gt_rgb, (render_res, render_res), interpolation=cv2.INTER_AREA)

    # ---------------- 전경 바운딩 박스 크롭 ----------------
    # 예측과 GT 전경의 합집합으로 잡아 늘어짐 아티팩트도 평가에 포함
    fg = (gt_rgb.sum(axis=2) > 0.02) | (render_np.sum(axis=2) > 0.02)
    ys, xs = np.where(fg)
    y0 = max(int(ys.min()) - crop_margin, 0)
    y1 = min(int(ys.max()) + 1 + crop_margin, render_res)
    x0 = max(int(xs.min()) - crop_margin, 0)
    x1 = min(int(xs.max()) + 1 + crop_margin, render_res)

    pred_crop = np.ascontiguousarray(render_np[y0:y1, x0:x1])
    gt_crop   = np.ascontiguousarray(gt_rgb[y0:y1, x0:x1])

    psnr    = compute_psnr(pred_crop, gt_crop)
    ssim    = compute_ssim(pred_crop, gt_crop)
    lpips_v = compute_lpips(pred_crop, gt_crop)

    if verbose:
        print("\n=========== THuman 2.0 Evaluation ===========")
        print(f" subject   : {subject}   ({src_view} -> {tgt_view})")
        print(f" angle     : {ang_rot:.2f}° (rotation) / {ang_orbit:.2f}° (orbit)")
        print(f" crop      : {y1-y0} x {x1-x0}  (from {render_res} x {render_res})")
        print(f" PSNR      : {psnr:.3f} dB")
        if ssim    is not None: print(f" SSIM      : {ssim:.4f}")
        if lpips_v is not None: print(f" LPIPS     : {lpips_v:.4f}")
        print("=============================================\n")

    # ---------------- 비교 이미지 저장 (크롭본) ----------------
    src_vis  = cv2.resize(src_bgr, (render_res, render_res))[y0:y1, x0:x1]
    pred_vis = cv2.cvtColor((pred_crop * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    gt_vis   = cv2.cvtColor((gt_crop  * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    combo = np.concatenate([src_vis, pred_vis, gt_vis], axis=1)
    combo = add_labels(combo, labels=("Input", "Ours", "Ground Truth"))
    cv2.imwrite(os.path.join(out_dir, f"{subject}_{src_view}to{tgt_view}.png"), combo)

    return {'subject': subject, 'src': src_view, 'tgt': tgt_view,
            'angle': ang_orbit, 'angle_rot': ang_rot,
            'psnr': psnr, 'ssim': ssim, 'lpips': lpips_v}


# --------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_ROOT  = "val"
    SUBJECT    = "0003_000"
    ENGINE_DIR = "trt_engines"
    OUT_DIR    = "eval_results"

    parm_dir = os.path.join(DATA_ROOT, "parm", SUBJECT)
    views    = list_views(parm_dir)
    _, e0    = load_cam(parm_dir, views[0])

    print(f"[INFO] available views: {views}")
    for v in views:
        _, ev = load_cam(parm_dir, v)
        print(f"  0 -> {v}: rot={view_angle(e0, ev):5.2f}°  orbit={orbit_angle(e0, ev):5.2f}°")

    # 엔진 1회 로드 후 전체 뷰 스윕
    print("\n[INFO] loading TensorRT engines...")
    engines = {
        'vda':  TRTWrapper(os.path.join(ENGINE_DIR, "vda_model.engine")),
        'unet': TRTWrapper(os.path.join(ENGINE_DIR, "unet_extractor.engine")),
        'gs':   TRTWrapper(os.path.join(ENGINE_DIR, "gs_regresser.engine")),
    }

    results = []
    for v in views:
        results.append(eval_thuman(
            data_root=DATA_ROOT, subject=SUBJECT,
            src_view=views[0], tgt_view=v,
            engine_dir=ENGINE_DIR, out_dir=OUT_DIR,
            warmup=32, engines=engines, verbose=False,
        ))

    results.sort(key=lambda r: r['angle'])
    print("\n================ Angle Sweep ================")
    print(f"{'view':>5} {'angle(°)':>9} {'PSNR(dB)':>9} {'SSIM':>7} {'LPIPS':>7}")
    for r in results:
        ssim_s  = f"{r['ssim']:.4f}"  if r['ssim']  is not None else "  -  "
        lpips_s = f"{r['lpips']:.4f}" if r['lpips'] is not None else "  -  "
        print(f"{r['tgt']:>5} {r['angle']:>9.2f} {r['psnr']:>9.3f} {ssim_s:>7} {lpips_s:>7}")
    print("=============================================\n")
    # ---- 깊이 스케일 진단 (고정 각도에서 배율만 변경) ----
    TGT = 1                                  # 4.30° 뷰
    print("\n============ Depth Scale Sweep ============")
    print(f"{'scale':>7} {'PSNR(dB)':>9} {'LPIPS':>8}")
    for s in [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]:
        r = eval_thuman(
            data_root=DATA_ROOT, subject=SUBJECT,
            src_view=views[0], tgt_view=TGT,
            engine_dir=ENGINE_DIR, out_dir=OUT_DIR,
            warmup=32, engines=engines, verbose=False,
            depth_scale=s,
        )
        lp = f"{r['lpips']:.4f}" if r['lpips'] is not None else "  -  "
        print(f"{s:>7.2f} {r['psnr']:>9.3f} {lp:>8}")
    print("===========================================\n")
    
