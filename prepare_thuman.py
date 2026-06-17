"""
THuman 2.0 데이터 폴더 구조 재구성 스크립트

실행 전 데이터 구조:
E:\THuman2.0\
    THuman2.0_obj\
        0000\ (0000.obj, material0.jpeg, material0.mtl)
        0001\
        ...
        0524\
    THuman2.0_smplx\
        0000\ (smplx_param.pkl)
        ...
        0524\

실행 후 데이터 구조 (render_data.py 호환):
E:\THuman2.0\
    train\        (0100 ~ 0524, 425개)
    val\          (0000 ~ 0099, 100개)
    THuman2.0_Smpl_X_Paras\
        0000\ (smplx_param.pkl)
        ...
"""

import os
import shutil
from tqdm import tqdm

# ── 경로 설정 ──────────────────────────────────────────────
THUMAN_ROOT  = r'E:\THuman2.0'
OBJ_ROOT     = os.path.join(THUMAN_ROOT, 'THuman2.0_obj')
SMPLX_ROOT   = os.path.join(THUMAN_ROOT, 'THuman2.0_smplx')
SMPLX_TARGET = os.path.join(THUMAN_ROOT, 'THuman2.0_Smpl_X_Paras')

# ── train / val 분할 (GPS-Gaussian 논문 기준) ──────────────
all_ids = sorted([f'{i:04d}' for i in range(525)])
val_ids   = [id for id in all_ids if int(id) < 100]    # 0000~0099 (100개)
train_ids = [id for id in all_ids if int(id) >= 100]   # 0100~0524 (425개)

print(f"train: {len(train_ids)}개  ({train_ids[0]} ~ {train_ids[-1]})")
print(f"val:   {len(val_ids)}개  ({val_ids[0]} ~ {val_ids[-1]})")
print()

# ── 1단계: OBJ 폴더 → train / val 구조로 복사 ─────────────
for phase, ids in [('train', train_ids), ('val', val_ids)]:
    print(f"[{phase}] OBJ 복사 중...")
    for data_id in tqdm(ids):
        src = os.path.join(OBJ_ROOT, data_id)
        dst = os.path.join(THUMAN_ROOT, phase, data_id)
        if os.path.exists(dst):
            continue
        shutil.copytree(src, dst)

# ── 2단계: SMPL-X → THuman2.0_Smpl_X_Paras 구조로 복사 ────
print("\n[SMPL-X] 파라미터 복사 중...")
os.makedirs(SMPLX_TARGET, exist_ok=True)
for data_id in tqdm(all_ids):
    src = os.path.join(SMPLX_ROOT, data_id, 'smplx_param.pkl')
    dst_dir = os.path.join(SMPLX_TARGET, data_id)
    dst = os.path.join(dst_dir, 'smplx_param.pkl')
    if not os.path.exists(src):
        print(f"  ⚠️  {data_id}: smplx_param.pkl 없음, 건너뜀")
        continue
    if os.path.exists(dst):
        continue
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, dst)

# ── 최종 구조 확인 ─────────────────────────────────────────
print("\n구조 확인:")
for phase in ['train', 'val']:
    path = os.path.join(THUMAN_ROOT, phase)
    if os.path.exists(path):
        count = len(os.listdir(path))
        print(f"  {phase}/: {count}개")
    else:
        print(f"  {phase}/: 없음")

smplx_count = len(os.listdir(SMPLX_TARGET)) if os.path.exists(SMPLX_TARGET) else 0
print(f"  THuman2.0_Smpl_X_Paras/: {smplx_count}개")
print("\n완료! render_data.py 실행 가능 상태입니다.")
