r"""
eval_stereo_compare_with_raft_trt.py
=====================================
RAFT TRT / IGEV++ TRT / Selective-IGEV TRT 통합 평가.
모든 모델을 TRT FP16으로 공정하게 비교.

실행 환경: IGEV_plusplus (TRT 8.6.x + pycuda)

기본 사용법 (sceneflow 파인튜닝 엔진 자동 탐색):
  cd C:\Users\User\Desktop\python\depth_estimation
  conda activate IGEV_plusplus

  python eval_stereo_compare_with_raft_trt.py ^
      --config config/stage1_scratch_bs2.yaml ^
      --num_samples 100

커스텀 엔진 경로 지정:
  python eval_stereo_compare_with_raft_trt.py ^
      --config config/stage1_scratch_bs2.yaml ^
      --raft_engine_i8 C:/path/to/raft_iters8.engine ^
      --igev_engine_i8 C:/path/to/igev_iters8.engine ^
      --sel_engine_i8  C:/path/to/sel_iters8.engine ^
      --num_samples 100

루트 경로 변경:
  --raft_root  RAFT-Stereo 폴더  (기본: ../RAFT-Stereo)
  --igev_root  IGEV++ 폴더       (기본: ../IGEV-plusplus)
  --sel_root   Selective-IGEV 폴더 (기본: ../Selective-Stereo/Selective-IGEV)
"""

import sys
import os
import time
import argparse
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── 인수 파싱 ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

# 필수
parser.add_argument('--config', required=True,
                    help='stage config YAML 경로')

# 루트 경로
parser.add_argument('--raft_root', default=None,
                    help='RAFT-Stereo 폴더 (기본: ../RAFT-Stereo)')
parser.add_argument('--igev_root', default=None,
                    help='IGEV++ 폴더 (기본: ../IGEV-plusplus)')
parser.add_argument('--sel_root',  default=None,
                    help='Selective-IGEV 폴더 (기본: ../Selective-Stereo/Selective-IGEV)')

# 개별 엔진 경로 (지정 시 루트+기본파일명보다 우선)
parser.add_argument('--raft_engine_i8', default=None)
parser.add_argument('--raft_engine_i4', default=None)
parser.add_argument('--igev_engine_i8', default=None)
parser.add_argument('--igev_engine_i4', default=None)
parser.add_argument('--sel_engine_i8',  default=None)
parser.add_argument('--sel_engine_i4',  default=None)

# 평가 설정
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--data_root',   default=None,
                    help='THuman 데이터 루트 (지정 시 config 값 덮어씀)')
parser.add_argument('--output_csv',  default='stereo_compare_all_trt_results.csv')
parser.add_argument('--no_raft',      action='store_true', help='RAFT-Stereo 평가 스킵')
parser.add_argument('--no_igev',      action='store_true', help='IGEV++ 평가 스킵')
parser.add_argument('--no_selective', action='store_true', help='Selective-IGEV 평가 스킵')
parser.add_argument('--vis_samples',  type=int, default=0,
                    help='시각화할 샘플 수 (0이면 스킵). 엔진당 앞 N개 샘플 저장')
parser.add_argument('--vis_dir',      default='disp_vis',
                    help='시각화 이미지 저장 폴더 (기본: disp_vis)')

args = parser.parse_args()

# ── 루트 경로 결정 ────────────────────────────────────────────────────
_BASE     = os.path.dirname(ROOT)
RAFT_ROOT = args.raft_root or os.path.join(_BASE, 'RAFT-Stereo')
IGEV_ROOT = args.igev_root or os.path.join(_BASE, 'IGEV-plusplus')
SEL_ROOT  = args.sel_root  or os.path.join(_BASE, 'Selective-Stereo', 'Selective-IGEV')

# IGEV core를 sys.path에 추가 (TRTStereoWrapper import 후 불필요하지만 안전하게 유지)
sys.path.insert(0, os.path.join(IGEV_ROOT, 'core'))
sys.path.insert(0, IGEV_ROOT)

def _resolve(explicit, root, default_name):
    """명시적 경로 우선, 없으면 root/default_name; 파일 없으면 None."""
    if explicit:
        return explicit
    p = os.path.join(root, default_name)
    return p if os.path.exists(p) else None

# SceneFlow 파인튜닝 엔진 파일명 (handover_v5 기준)
raft_engine_i8 = _resolve(args.raft_engine_i8, RAFT_ROOT, 'raft_sceneflow_iters8_fp16.engine')
raft_engine_i4 = _resolve(args.raft_engine_i4, RAFT_ROOT, 'raft_sceneflow_iters4_fp16.engine')
igev_engine_i8 = _resolve(args.igev_engine_i8, IGEV_ROOT, 'igev_sceneflow_iters8_fp16.engine')
igev_engine_i4 = _resolve(args.igev_engine_i4, IGEV_ROOT, 'igev_sceneflow_iters4_fp16.engine')
sel_engine_i8  = _resolve(args.sel_engine_i8,  SEL_ROOT,  'selective_sceneflow_iters8_fp16.engine')
sel_engine_i4  = _resolve(args.sel_engine_i4,  SEL_ROOT,  'selective_sceneflow_iters4_fp16.engine')

# ── 엔진 경로 사전 확인 ────────────────────────────────────────────────
print("=" * 70)
print("엔진 경로 확인")
print("-" * 70)
for label, path in [
    ('RAFT  i8', raft_engine_i8), ('RAFT  i4', raft_engine_i4),
    ('IGEV  i8', igev_engine_i8), ('IGEV  i4', igev_engine_i4),
    ('SEL   i8', sel_engine_i8),  ('SEL   i4', sel_engine_i4),
]:
    if path and os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {label}: {os.path.basename(path)}  ({size_mb:.1f} MB)  OK")
    elif path:
        print(f"  {label}: {path}  !! 파일 없음 - 건너뜀")
    else:
        print(f"  {label}: (미지정 - 건너뜀)")
print("=" * 70)

from lib.human_loader import StereoHumanDataset
from trt_stereo_wrapper import TRTStereoWrapper
from config.stereo_human_config import ConfigStereoHuman

# ── 시각화 헬퍼 ───────────────────────────────────────────────────────
def tensor_to_rgb(img_tensor):
    img = (img_tensor.squeeze(0).permute(1,2,0).cpu().numpy() + 1.0) / 2.0
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def disp_to_colormap(disp_np, vmin=None, vmax=None):
    valid = np.isfinite(disp_np)
    if vmin is None: vmin = np.nanpercentile(disp_np, 2)
    if vmax is None: vmax = np.nanpercentile(disp_np, 98)
    norm  = np.clip((disp_np - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colored = (plt.get_cmap('plasma')(norm)[:, :, :3] * 255).astype(np.uint8)
    colored[~valid] = 0
    return colored, vmin, vmax

def error_to_colormap(epe_np, vmax=None):
    valid = np.isfinite(epe_np)
    if vmax is None: vmax = np.nanpercentile(epe_np, 95)
    norm  = np.clip(epe_np / (vmax + 1e-8), 0, 1)
    colored = (plt.get_cmap('hot')(norm)[:, :, :3] * 255).astype(np.uint8)
    colored[~valid] = 0
    return colored, vmax

def save_vis(left, disp_pred, flow_gt, valid, save_path, epe, d1):
    gt_np     = flow_gt.float().squeeze().cpu().numpy()
    pred_np   = disp_pred.float().squeeze().cpu().numpy()
    valid_np  = (valid.float().squeeze().cpu().numpy() >= 0.5)

    # 배경 마스킹
    gt_masked   = gt_np.copy();   gt_masked[~valid_np]   = np.nan
    pred_masked = pred_np.copy(); pred_masked[~valid_np] = np.nan
    epe_np      = np.abs(pred_np - gt_np); epe_np[~valid_np] = np.nan

    vmin = np.nanpercentile(gt_masked, 2)
    vmax = np.nanpercentile(gt_masked, 98)

    left_rgb,            = [tensor_to_rgb(left)]
    gt_color,   _, _     = disp_to_colormap(gt_masked,   vmin=vmin, vmax=vmax)
    pred_color, _, _     = disp_to_colormap(pred_masked, vmin=vmin, vmax=vmax)
    err_color,  emax     = error_to_colormap(epe_np)

    fig = plt.figure(figsize=(20, 5))
    gs  = gridspec.GridSpec(1, 4, wspace=0.03)
    titles = [
        'Left image',
        f'GT disparity\n(min={vmin:.1f} max={vmax:.1f})',
        f'Pred disparity\n(EPE={epe:.4f}  D1={d1:.2f}%)',
        f'Error map (|pred-GT|)\n(max={emax:.2f}px)',
    ]
    for j, (im, title) in enumerate(zip([left_rgb, gt_color, pred_color, err_color], titles)):
        ax = fig.add_subplot(gs[j])
        ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close(fig)

# ── config / dataloader ───────────────────────────────────────────────
cfg_parser = ConfigStereoHuman()
cfg_parser.load(args.config)
cfg = cfg_parser.get_cfg()
if args.data_root:
    cfg.defrost()
    cfg.dataset.data_root = args.data_root
    cfg.freeze()

val_set    = StereoHumanDataset(cfg.dataset, phase='val')
val_loader = DataLoader(val_set, batch_size=1, num_workers=0, shuffle=True)
print(f"Val 샘플 수: {len(val_set)}  (평가: {args.num_samples}개)")

# ── 메트릭 ───────────────────────────────────────────────────────────
def compute_metrics(flow_pred, flow_gt, valid):
    # float16 overflow 방지: 반드시 float32로 변환
    flow_pred = flow_pred.float().view(-1)
    flow_gt   = flow_gt.float().view(-1)
    valid     = valid.float().view(-1)

    mask = (valid >= 0.5) & torch.isfinite(flow_gt) & torch.isfinite(flow_pred)
    if mask.sum() == 0:
        return None, None

    epe_v = (flow_pred[mask] - flow_gt[mask]).abs()
    gt_v  = flow_gt[mask].abs()
    epe_mean = epe_v.mean().item()
    d1 = ((epe_v > 1) & (epe_v / (gt_v + 1e-8) > 0.05)).float().mean().item() * 100
    return epe_mean, d1

# ── TRT 평가 ─────────────────────────────────────────────────────────
def evaluate_trt(engine_path, val_loader, num_samples, model_name,
                 vis_samples=0, vis_dir=None):
    """lmain 단독 평가 (학습도 lmain 기준으로만 했으므로 동일하게 맞춤)."""
    wrapper = TRTStereoWrapper(engine_path)
    epe_list, d1_list, time_list = [], [], []
    skipped = 0

    # 시각화 폴더 준비
    if vis_samples > 0 and vis_dir:
        eng_dir = os.path.join(vis_dir, os.path.splitext(os.path.basename(engine_path))[0])
        os.makedirs(eng_dir, exist_ok=True)
    else:
        eng_dir = None

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= num_samples:
                break

            left    = data['lmain']['img'].cuda()
            right   = data['rmain']['img'].cuda()
            flow_gt = data['lmain']['flow'].cuda()
            valid   = data['lmain']['valid'].cuda()

            t0   = time.perf_counter()
            disp = wrapper.infer(left, right)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            epe, d1 = compute_metrics(disp, flow_gt, valid)
            if epe is None:
                skipped += 1
            else:
                epe_list.append(epe)
                d1_list.append(d1)
                time_list.append(elapsed_ms)

                # 시각화 저장
                if eng_dir and i < vis_samples:
                    save_path = os.path.join(eng_dir, f'sample{i:03d}_epe{epe:.4f}_d1{d1:.2f}.png')
                    save_vis(left, disp, flow_gt, valid, save_path, epe, d1)

    del wrapper
    torch.cuda.empty_cache()

    if not epe_list:
        print(f"  !! 유효 샘플 없음 (skipped={skipped})")
        return None, None, None

    print(f"  (유효 {len(epe_list)}개, 건너뜀 {skipped}개)")
    return np.mean(epe_list), np.mean(d1_list), np.mean(time_list)

# ── 평가 실행 ─────────────────────────────────────────────────────────
engine_configs = []
if not args.no_raft:
    engine_configs += [
        ('RAFT-Stereo TRT', 8, raft_engine_i8),
        ('RAFT-Stereo TRT', 4, raft_engine_i4),
    ]
if not args.no_igev:
    engine_configs += [
        ('IGEV++ TRT',      8, igev_engine_i8),
        ('IGEV++ TRT',      4, igev_engine_i4),
    ]
if not args.no_selective:
    engine_configs += [
        ('Selective TRT',   8, sel_engine_i8),
        ('Selective TRT',   4, sel_engine_i4),
    ]

results = []
for model_name, iters, engine_path in engine_configs:
    if not engine_path or not os.path.exists(engine_path):
        continue
    print(f"\n[{model_name} iters={iters}] 평가 중...")
    epe, d1, ms = evaluate_trt(engine_path, val_loader, args.num_samples, model_name,
                               vis_samples=args.vis_samples, vis_dir=args.vis_dir)
    if epe is None:
        continue
    results.append((model_name, iters, epe, d1, ms))
    print(f"  EPE={epe:.4f}  D1={d1:.2f}%  Time={ms:.1f}ms")

# ── 결과 출력 ─────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print(f"{'모델':<24} {'iters':>5}  {'EPE':>8}  {'D1(%)':>8}  {'Time(ms)':>10}")
print("-" * 68)
for model, iters, epe, d1, ms in results:
    mark = " *" if model == 'RAFT-Stereo TRT' and iters == 8 else ""
    print(f"{model:<24} {iters:>5}  {epe:>8.4f}  {d1:>8.2f}  {ms:>10.1f}{mark}")
print("=" * 68)

with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'iters', 'EPE', 'D1(%)', 'Time(ms)'])
    for row in results:
        writer.writerow(row)
print(f"\n결과 저장: {args.output_csv}")
