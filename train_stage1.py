import os
import sys
import logging
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from config.stereo_human_config import ConfigStereoHuman
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from lib.train_recoder import Logger, file_backup

# ------------------------------------------------------------------ #
#  Stage 1: RAFT-Stereo disparity 학습만 수행 (GS 렌더링 없음)
#  실행 예시:
#    python train_stage1.py --config config/stage1.yaml
# ------------------------------------------------------------------ #


def fetch_optimizer(cfg, model):
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wdecay, eps=1e-8)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        cfg.lr,
        cfg.num_steps + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy='linear'
    )
    return optimizer, scheduler


def validate(model, val_loader, total_steps):
    model.eval()
    epe_list, d1_list = [], []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= 50:   # val은 50 배치만 샘플링
                break
            data['lmain']['img']  = data['lmain']['img'].cuda()
            data['rmain']['img']  = data['rmain']['img'].cuda()
            data['lmain']['flow'] = data['lmain']['flow'].cuda()
            data['rmain']['flow'] = data['rmain']['flow'].cuda()
            data['lmain']['valid'] = data['lmain']['valid'].cuda()
            data['rmain']['valid'] = data['rmain']['valid'].cuda()
            data['lmain']['mask']  = data['lmain']['mask'].cuda()
            data['rmain']['mask']  = data['rmain']['mask'].cuda()
            data['lmain']['intr']  = data['lmain']['intr'].cuda()
            data['rmain']['intr']  = data['rmain']['intr'].cuda()
            data['lmain']['ref_intr'] = data['lmain']['ref_intr'].cuda()
            data['rmain']['ref_intr'] = data['rmain']['ref_intr'].cuda()
            data['lmain']['extr']  = data['lmain']['extr'].cuda()
            data['rmain']['extr']  = data['rmain']['extr'].cuda()
            data['lmain']['Tf_x']  = data['lmain']['Tf_x'].cuda()
            data['rmain']['Tf_x']  = data['rmain']['Tf_x'].cuda()

            data, flow_loss, metrics = model(data, is_train=False)

            # flow_pred vs flow_gt → EPE, D1
            for view in ['lmain', 'rmain']:
                pred  = data[view]['flow_pred']
                gt    = data[view]['flow'].cuda()
                valid = data[view]['valid'].cuda() >= 0.5

                epe = (pred - gt).abs().squeeze(1)
                epe_list.append(epe[valid.squeeze(1)].mean().item())

                d1 = ((epe > 1) & ((epe / (gt.abs().squeeze(1) + 1e-8)) > 0.05))
                d1_list.append(d1[valid.squeeze(1)].float().mean().item())

    mean_epe = np.mean(epe_list)
    mean_d1  = np.mean(d1_list) * 100
    logging.info(f"[Val {total_steps}] EPE: {mean_epe:.4f}  D1: {mean_d1:.2f}%")
    model.train()
    return {'val_epe': mean_epe, 'val_d1': mean_d1}


def move_data_to_cuda(data):
    keys = ['img', 'flow', 'valid', 'mask', 'intr', 'ref_intr', 'extr', 'Tf_x']
    for view in ['lmain', 'rmain']:
        for k in keys:
            if k in data[view]:
                data[view][k] = data[view][k].cuda()
    return data


def train(cfg_file):
    # ---------- config ----------
    cfg_parser = ConfigStereoHuman()
    cfg_parser.load(cfg_file)
    cfg = cfg_parser.get_cfg()

    # ---------- output 경로 ----------
    exp_name = cfg.name
    ckpt_path = os.path.join('output', exp_name, 'checkpoints')
    logs_path = os.path.join('output', exp_name, 'logs')
    show_path = os.path.join('output', exp_name, 'show')
    file_path = os.path.join('output', exp_name, 'train_log.txt')
    for p in [ckpt_path, logs_path, show_path]:
        os.makedirs(p, exist_ok=True)

    # cfg에 경로 주입
    cfg.defrost()
    cfg.record.ckpt_path  = ckpt_path
    cfg.record.logs_path  = logs_path
    cfg.record.show_path  = show_path
    cfg.record.file_path  = file_path
    cfg.freeze()

    # ---------- logging ----------
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Config: {cfg_file}")
    logging.info(f"data_root: {cfg.dataset.data_root}")
    logging.info(f"train_iters: {cfg.raft.train_iters}  val_iters: {cfg.raft.val_iters}")

    # ---------- 코드 백업 ----------
    file_backup(os.path.join('output', exp_name), dict(cfg), __file__)

    # ---------- dataset ----------
    train_set = StereoHumanDataset(cfg.dataset, phase='train')
    val_set   = StereoHumanDataset(cfg.dataset, phase='val')
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        num_workers=2,
        shuffle=False
    )
    logging.info(f"Train samples: {len(train_set)}  Val samples: {len(val_set)}")

    # ---------- model ----------
    model = RtStereoHumanModel(cfg, with_gs_render=False).cuda()
    model.train()

    if cfg.restore_ckpt is not None and cfg.restore_ckpt != 'None':
        ckpt = torch.load(cfg.restore_ckpt, map_location='cuda')
        model.load_state_dict(ckpt['model'], strict=False)
        logging.info(f"Restored from {cfg.restore_ckpt}")

    # ---------- optimizer ----------
    optimizer, scheduler = fetch_optimizer(cfg, model)
    scaler   = GradScaler(enabled=cfg.raft.mixed_precision)
    logger   = Logger(scheduler, cfg.record)

    total_steps = 0
    train_iter  = iter(train_loader)

    logging.info("===== Stage 1 Training Start =====")

    while total_steps <= cfg.num_steps:
        # --- data ---
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data = next(train_iter)

        data = move_data_to_cuda(data)

        # --- forward ---
        optimizer.zero_grad()
        data, flow_loss, metrics = model(data, is_train=True)

        # --- backward ---
        scaler.scale(flow_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        logger.push(metrics)

        # --- val & save ---
        if total_steps > 0 and total_steps % cfg.record.eval_freq == 0:
            val_metrics = validate(model, val_loader, total_steps)
            logger.write_dict(val_metrics, total_steps)

            ckpt_file = os.path.join(ckpt_path, f'stage1_{total_steps:06d}.pth')
            torch.save({'model': model.state_dict(), 'steps': total_steps}, ckpt_file)
            logging.info(f"Saved checkpoint: {ckpt_file}")

        total_steps += 1

    # ---------- final save ----------
    final_ckpt = os.path.join(ckpt_path, 'GPS-GS_stage1_final.pth')
    torch.save({'model': model.state_dict(), 'steps': total_steps}, final_ckpt)
    logging.info(f"Training Done. Final ckpt: {final_ckpt}")
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to stage1 yaml config')
    args = parser.parse_args()
    train(args.config)
