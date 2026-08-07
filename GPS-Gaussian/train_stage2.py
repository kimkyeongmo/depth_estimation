from __future__ import print_function, division

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import numpy as np
import cv2
import subprocess
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import lpips 

from lib.human_loader import StereoHumanDataset
from lib.network import VDAGaussianModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.GaussianRender import pts2render
from lib.loss import l1_loss, ssim, psnr

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

cv2.setNumThreads(0)
torch.set_float32_matmul_precision('high')

class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file
        self.model = VDAGaussianModel(self.cfg, with_gs_render=True)

        self.train_set = StereoHumanDataset(self.cfg.dataset, phase='train')
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=2 * self.cfg.batch_size, pin_memory=True,
            drop_last=True, persistent_workers=True
        )
        self.train_iterator = iter(self.train_loader)

        self.val_set = StereoHumanDataset(self.cfg.dataset, phase='val')
        self.val_loader = DataLoader(
            self.val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
        )
        self.len_val = len(self.val_loader)
        self.val_iterator = iter(self.val_loader)

        self.total_steps = 0
        self.model.cuda()

        logging.info("Loading Checkpoints for True Fine-tuning...")
        resume_ckpt_path = 'checkpoints/base/GPS-GS_stage2_final.pth'

        if os.path.exists(resume_ckpt_path):
            ckpt = torch.load(resume_ckpt_path, map_location='cuda', weights_only=False)
            self.model.load_state_dict(ckpt['network'], strict=False)
            self.total_steps = 0
            logging.info("Vanilla knowledge loaded via strict=False.")
        else:
            logging.warning(f"Checkpoint not found at {resume_ckpt_path}. Starting from scratch.")

        for param in self.model.vda_model.parameters():
            param.requires_grad = False

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optim.AdamW(trainable_params, lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, self.cfg.lr, self.cfg.num_steps + 100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear'
        )

        self.logger = Logger(self.scheduler, cfg.record)
        self.model.train()
        self.model.vda_model.eval()

        self.best_psnr = -float('inf')
        
        self.lpips_metric = lpips.LPIPS(net='vgg').cuda()
        self.lpips_metric.requires_grad_(False)
        
        self._nan_streak = 0
        self._nan_streak_limit = 5

    def _set_train_mode(self):
        self.model.train()
        self.model.vda_model.eval()

    def _reset_optimizer_state(self):
        logging.warning(f"[Step {self.total_steps}] Resetting optimizer momentum (NaN streak).")
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state:
                    s = self.optimizer.state[p]
                    if 'exp_avg'    in s: s['exp_avg'].zero_()
                    if 'exp_avg_sq' in s: s['exp_avg_sq'].zero_()
        self._nan_streak = 0

    def train(self):
        def to_float32_and_contiguous(x):
            if isinstance(x, dict):
                return {k: to_float32_and_contiguous(v) for k, v in x.items()}
            elif isinstance(x, torch.Tensor) and x.is_floating_point():
                return x.float().contiguous()
            return x

        logging.info(f"Training from step {self.total_steps} to {self.cfg.num_steps}. VDA is frozen. float32 precision loaded.")

        for _ in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad(set_to_none=True)
            data = self.fetch_data(phase='train')

            data, _, _ = self.model(data, is_train=True)
            data = to_float32_and_contiguous(data)
            
            # 🚀 불필요한 isfinite 체크 루프 제거, 마스크 연산 최소화
            if 'view_0' in data and 'pts_valid' in data['view_0']:
                valid_mask = data['view_0']['pts_valid']
                depth_flat = data['view_0']['depth'].view(valid_mask.shape)
                valid_mask = valid_mask & (depth_flat > 0.3)

                if 'mask' in data['view_0']:
                    mask_flat = data['view_0']['mask'].view(valid_mask.shape)
                    valid_mask = valid_mask & (mask_flat > 0.5)

                data['view_0']['pts_valid'] = valid_mask

            total_loss = 0.0
            total_l1 = 0.0
            total_ssim = 0.0
            
            # 💡 0번(자기 재구성) 및 1, 2, 3, 4번 노벨 뷰 모두 렌더링 및 손실 계산
            target_indices = [0, 1, 2, 3, 4]
            valid_views = 0

            for view_idx in target_indices:
                view_key = f'view_{view_idx}'
                if view_key not in data: 
                    continue
                
                # 기존 GaussianRender.py의 인터페이스를 유지하기 위한 딕셔너리 트릭
                render_dict = {
                    'lmain': data['view_0'],
                    'novel_view': data[view_key]
                }
                
                render_out = pts2render(render_dict, bg_color=self.cfg.dataset.bg_color)
                render_img = render_out['novel_view']['img_pred']
                gt_img = data[view_key]['img']

                if 'mask' in data[view_key]:
                    mask = data[view_key]['mask']
                    gt_clean = gt_img * mask
                    soft_mask = mask + 0.1 * (1.0 - mask)

                    render_soft = render_img * soft_mask
                    gt_soft = gt_clean * soft_mask

                    l1_val = torch.abs(render_soft - gt_soft).mean()
                    ssim_val = 1.0 - ssim(render_soft, gt_soft)
                else:
                    l1_val = l1_loss(render_img, gt_img)
                    ssim_val = 1.0 - ssim(render_img, gt_img)

                total_l1 += l1_val
                total_ssim += ssim_val
                valid_views += 1
            
            if valid_views == 0:
                continue
                
            avg_l1 = total_l1 / valid_views
            avg_ssim = total_ssim / valid_views
            loss = 0.90 * avg_l1 + 0.10 * avg_ssim

            if not torch.isfinite(loss):
                self._nan_streak += 1
                logging.warning(f"[Step {self.total_steps}] NaN loss! streak={self._nan_streak}.")
                if self._nan_streak >= self._nan_streak_limit:
                    self._reset_optimizer_state()
                self.optimizer.zero_grad(set_to_none=True)
                self.total_steps += 1
                continue
            else:
                self._nan_streak = 0

            loss.backward()
            
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            self.optimizer.step()
            self.scheduler.step()

            log_freq = max(1, self.cfg.record.eval_freq // 10)
            if self.total_steps % log_freq == 0:
                self.logger.write_dict({
                    'loss':         loss.item(),
                    'loss_l1':      avg_l1.item(),
                    'loss_ssim':    avg_ssim.item(),
                    'grad_norm':    grad_norm.item(),
                    'lr_base':      self.optimizer.param_groups[0]['lr'],
                }, write_step=self.total_steps)

            if self.total_steps > 0 and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                val_psnr = self.run_eval()

                save_name = os.path.join(self.cfg.record.ckpt_path, f"{self.total_steps}.pth")
                self.save_ckpt(save_path=save_name, show_log=False)

                if val_psnr > self.best_psnr:
                    self.best_psnr = val_psnr
                    best_path = os.path.join(self.cfg.record.ckpt_path, "best.pth")
                    self.save_ckpt(save_path=best_path, show_log=False)
                    logging.info(f"[Step {self.total_steps}] New best True PSNR={val_psnr:.4f} → {best_path}")

                self._set_train_mode()

            self.total_steps += 1

        logging.info("Training finished.")
        final_path = Path('%s/%s_final.pth' % (self.cfg.record.ckpt_path, self.cfg.exp_name))
        self.save_ckpt(save_path=final_path)
        self.upload_to_drive()

    def upload_to_drive(self):
        local_path  = os.path.dirname(self.cfg.record.ckpt_path)
        remote_path = f"gdrive:VDA-Gaussian/{self.cfg.exp_name}"
        logging.info(f"Uploading {local_path} → {remote_path} via rclone...")
        try:
            subprocess.run(["rclone", "copy", local_path, remote_path], check=True)
            logging.info("Upload completed.")
        except Exception as e:
            logging.error(f"Upload failed: {e}")

    def run_eval(self):
        logging.info(f"[Step {self.total_steps}] Running validation ...")
        torch.cuda.empty_cache()

        psnr_list, ssim_list, lpips_list = [], [], []
        show_idx = np.random.choice(list(range(self.len_val)), 1)

        for idx in range(self.len_val):
            data = self.fetch_data(phase='val')
            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)

                # 평가 단계에서도 5개의 시점에 대해 렌더링 품질을 모두 검증
                for view_idx in [0, 1, 2, 3, 4]:
                    view_key = f'view_{view_idx}'
                    if view_key not in data: continue
                    
                    render_dict = {'lmain': data['view_0'], 'novel_view': data[view_key]}
                    render_out = pts2render(render_dict, bg_color=self.cfg.dataset.bg_color)

                    render_novel = render_out['novel_view']['img_pred']
                    gt_novel = data[view_key]['img']

                    render_lpips = render_novel * 2.0 - 1.0
                    gt_lpips = gt_novel * 2.0 - 1.0

                    if 'mask' in data[view_key]:
                        mask = data[view_key]['mask']
                        psnr_val = psnr(render_novel * mask, gt_novel * mask, mask=mask).mean().item()
                        ssim_val = ssim(render_novel * mask, gt_novel * mask).mean().item()
                        lpips_val = self.lpips_metric((render_lpips * mask), (gt_lpips * mask)).mean().item()
                    else:
                        psnr_val = psnr(render_novel, gt_novel).mean().item()
                        ssim_val = ssim(render_novel, gt_novel).mean().item()
                        lpips_val = self.lpips_metric(render_lpips, gt_lpips).mean().item()

                    psnr_list.append(psnr_val)
                    ssim_list.append(ssim_val)
                    lpips_list.append(lpips_val)

                    if idx == show_idx and view_idx == 4:
                        tmp = (render_novel[0].detach() * 255).permute(1, 2, 0).cpu().numpy()
                        cv2.imwrite('%s/%s.jpg' % (self.cfg.record.show_path, self.total_steps), tmp[:, :, ::-1].astype(np.uint8))

        val_psnr  = float(np.round(np.mean(psnr_list), 4))
        val_ssim  = float(np.round(np.mean(ssim_list), 4))
        val_lpips = float(np.round(np.mean(lpips_list), 4))

        logging.info(f"[Step {self.total_steps}] Val — PSNR: {val_psnr} | SSIM: {val_ssim} | LPIPS: {val_lpips}")
        self.logger.write_dict({'val_psnr': val_psnr, 'val_ssim': val_ssim, 'val_lpips': val_lpips}, write_step=self.total_steps)

        torch.cuda.empty_cache()
        return val_psnr

    def fetch_data(self, phase):
        if phase == 'train':
            try: data = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        else:
            try: data = next(self.val_iterator)
            except StopIteration:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

        # 💡 새로운 데이터 로더 형식에 맞춰 view_0 ~ view_4로 순회 할당
        for view in ['view_0', 'view_1', 'view_2', 'view_3', 'view_4']:
            if view in data:
                for item in data[view].keys():
                    if isinstance(data[view][item], torch.Tensor):
                        data[view][item] = data[view][item].cuda()
        return data

    def save_ckpt(self, save_path, show_log=True):
        if show_log: logging.info(f"Saving checkpoint → {save_path}")
        torch.save({
            'total_steps': self.total_steps,
            'network':     self.model.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
            'scheduler':   self.scheduler.state_dict(),
            'best_psnr':   self.best_psnr,
        }, save_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load("config/stage2.yaml")
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.dataset.data_root = '../thuman_120cm_render_data_mono'

    dt = datetime.today()
    cfg.exp_name         = 'VDA_GPS_%s%s_Finetune' % (str(dt.month).zfill(2), str(dt.day).zfill(2))
    cfg.record.ckpt_path = "experiments/%s/ckpt"   % cfg.exp_name
    cfg.record.show_path = "experiments/%s/show"   % cfg.exp_name
    cfg.record.logs_path = "experiments/%s/logs"   % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file"   % cfg.exp_name
    cfg.freeze()

    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)

    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = Trainer(cfg)
    trainer.train()