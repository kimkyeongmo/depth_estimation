"""
regenerate_cache.py
────────────────────
버그 수정된 human_loader로 rectified_local 캐시를 재생성합니다.

수정 내용:
  - 버그①: stereoRectify flags=0 → CALIB_ZERO_DISPARITY (cx offset 제거)
  - 버그③: img * mask → 배경을 ImageNet mean으로 채움

실행 예시:
  cd C:\\Users\\User\\Desktop\\python\\depth_estimation
  conda activate IGEV_plusplus
  python regenerate_cache.py --data_root E:/THuman2.0/rendered_data_512
"""

import sys
import os
import argparse
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('--gps_root',  default='C:/Users/User/Desktop/python/depth_estimation')
parser.add_argument('--data_root', required=True)
args = parser.parse_args()

sys.path.insert(0, args.gps_root)

from yacs.config import CfgNode as CN
from lib.human_loader import StereoHumanDataset

def make_cfg(data_root, phase):
    cfg = CN()
    cfg.data_root          = data_root
    cfg.source_id          = [0, 1]
    cfg.src_res            = 512
    cfg.use_processed_data = True
    cfg.train_novel_id     = [2, 3, 4]
    cfg.val_novel_id       = [2, 3, 4]
    cfg.use_hr_img         = False
    cfg.force_regenerate   = True   # 기존 캐시 덮어쓰기
    return cfg

for phase in ['train', 'val']:
    logging.info(f"===== {phase} 캐시 재생성 시작 =====")
    cfg = make_cfg(args.data_root, phase)
    ds = StereoHumanDataset(cfg, phase=phase)
    logging.info(f"===== {phase} 완료 =====")

logging.info("전체 캐시 재생성 완료.")
logging.info("이제 finetune_igev_v4.py / finetune_selective_v4.py로 v5 학습을 시작하세요.")
