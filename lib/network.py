"""
lib/network.py (수정본)
========================
GPS-Gaussian Stage1 네트워크.
stereo_model 인자로 추론 백엔드를 선택할 수 있음:
  - 'raft'     : RAFT-Stereo (트레이닝/PyTorch 추론)
  - 'igev'     : IGEV++ TRT FP16 (추론 전용)
  - 'selective': Selective-IGEV TRT FP16 (추론 전용)

TRT 모드에서는 img_encoder / raft_stereo 대신
TRTStereoWrapper를 사용하여 disparity를 직접 계산.
"""

import torch
from torch import nn
import torch.nn.functional as F
from core.raft_stereo_human import RAFTStereoHuman
from core.extractor import UnetExtractor
from lib.gs_parm_network import GSRegresser
from lib.loss import sequence_loss
from lib.utils import flow2depth, depth2pc
from torch.cuda.amp import autocast as autocast


class RtStereoHumanModel(nn.Module):
    def __init__(self, cfg, with_gs_render=False, stereo_model='raft',
                 igev_engine=None, selective_engine=None):
        """
        Args:
            cfg            : YAML config
            with_gs_render : Gaussian 렌더링 포함 여부
            stereo_model   : 'raft' | 'igev' | 'selective'
            igev_engine    : IGEV++ .engine 파일 경로 (stereo_model='igev' 시 필요)
            selective_engine: Selective-IGEV .engine 파일 경로
        """
        super().__init__()
        self.cfg            = cfg
        self.with_gs_render = with_gs_render
        self.stereo_model   = stereo_model
        self.train_iters    = self.cfg.raft.train_iters
        self.val_iters      = self.cfg.raft.val_iters

        if stereo_model == 'raft':
            # 원본 RAFT-Stereo 파이프라인
            self.img_encoder  = UnetExtractor(in_channel=3, encoder_dim=self.cfg.raft.encoder_dims)
            self.raft_stereo  = RAFTStereoHuman(self.cfg.raft)
            self.trt_wrapper  = None
        else:
            # TRT 모드: img_encoder / raft_stereo 불필요
            self.img_encoder  = None
            self.raft_stereo  = None
            self._init_trt(stereo_model, igev_engine, selective_engine)

        if self.with_gs_render:
            self.gs_parm_regresser = GSRegresser(self.cfg, rgb_dim=3, depth_dim=1)

    def _init_trt(self, stereo_model, igev_engine, selective_engine):
        from trt_stereo_wrapper import TRTStereoWrapper
        if stereo_model == 'igev':
            assert igev_engine is not None, "--igev_engine 경로를 지정해주세요."
            self.trt_wrapper = TRTStereoWrapper(igev_engine)
        elif stereo_model == 'selective':
            assert selective_engine is not None, "--selective_engine 경로를 지정해주세요."
            self.trt_wrapper = TRTStereoWrapper(selective_engine)
        else:
            raise ValueError(f"알 수 없는 stereo_model: {stereo_model}")

    def forward(self, data, is_train=True):
        bs = data['lmain']['img'].shape[0]

        if self.stereo_model == 'raft':
            return self._forward_raft(data, is_train, bs)
        else:
            return self._forward_trt(data, bs)

    # ── RAFT-Stereo 경로 (원본 유지) ─────────────────────────────────────────
    def _forward_raft(self, data, is_train, bs):
        image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)
        flow  = torch.cat([data['lmain']['flow'], data['rmain']['flow']], dim=0) if is_train else None
        valid = torch.cat([data['lmain']['valid'], data['rmain']['valid']], dim=0) if is_train else None

        with autocast(enabled=self.cfg.raft.mixed_precision):
            img_feat = self.img_encoder(image)

        if is_train:
            flow_predictions = self.raft_stereo(img_feat[2], iters=self.train_iters)
            flow_loss, metrics = sequence_loss(flow_predictions, flow, valid)
            flow_pred_lmain, flow_pred_rmain = torch.split(flow_predictions[-1], [bs, bs])

            if not self.with_gs_render:
                data['lmain']['flow_pred'] = flow_pred_lmain.detach()
                data['rmain']['flow_pred'] = flow_pred_rmain.detach()
                return data, flow_loss, metrics

            data['lmain']['flow_pred'] = flow_pred_lmain
            data['rmain']['flow_pred'] = flow_pred_rmain
            data = self.flow2gsparms(image, img_feat, data, bs)
            return data, flow_loss, metrics

        else:
            flow_up = self.raft_stereo(img_feat[2], iters=self.val_iters, test_mode=True)
            flow_loss, metrics = None, None

            data['lmain']['flow_pred'] = flow_up[0]
            data['rmain']['flow_pred'] = flow_up[1]

            if not self.with_gs_render:
                return data, flow_loss, metrics
            data = self.flow2gsparms(image, img_feat, data, bs)
            return data, flow_loss, metrics

    # ── TRT 경로 (추론 전용) ─────────────────────────────────────────────────
    def _forward_trt(self, data, bs):
        """
        TRT 엔진으로 disparity 추론.
        트레이닝 불가 (추론 전용).

        data['lmain']['img']: (B, 3, H, W) [-1, 1]
        data['rmain']['img']: (B, 3, H, W) [-1, 1]

        출력:
        data['lmain']['flow_pred']: (B, 1, H, W) disparity
        data['rmain']['flow_pred']: (B, 1, H, W) disparity (부호 반전)
        """
        left  = data['lmain']['img']   # (B, 3, H, W) [-1,1]
        right = data['rmain']['img']   # (B, 3, H, W) [-1,1]

        with torch.no_grad():
            # lmain: left → right 방향 disparity
            disp_lr = self.trt_wrapper.infer(left, right)   # (B, 1, H, W)
            # rmain: right → left 방향 disparity (부호 반전)
            disp_rl = self.trt_wrapper.infer(right, left)   # (B, 1, H, W)

        data['lmain']['flow_pred'] = disp_lr
        data['rmain']['flow_pred'] = disp_rl

        flow_loss, metrics = None, None

        if not self.with_gs_render:
            return data, flow_loss, metrics

        # GS 렌더링은 img_feat 필요 → TRT 모드에서는 with_gs_render=False로만 사용
        raise NotImplementedError(
            "TRT 모드에서 with_gs_render=True는 지원하지 않습니다. "
            "eval_stereo_compare.py는 with_gs_render=False로 실행하세요."
        )

    def flow2gsparms(self, lr_img, lr_img_feat, data, bs):
        for view in ['lmain', 'rmain']:
            data[view]['depth'] = flow2depth(data[view])
            data[view]['xyz']   = depth2pc(
                data[view]['depth'], data[view]['extr'], data[view]['intr']
            ).view(bs, -1, 3)
            valid = data[view]['depth'] != 0.0
            data[view]['pts_valid'] = valid.view(bs, -1)

        lr_depth = torch.concat([data['lmain']['depth'], data['rmain']['depth']], dim=0)
        rot_maps, scale_maps, opacity_maps = self.gs_parm_regresser(lr_img, lr_depth, lr_img_feat)

        data['lmain']['rot_maps'],     data['rmain']['rot_maps']     = torch.split(rot_maps,     [bs, bs])
        data['lmain']['scale_maps'],   data['rmain']['scale_maps']   = torch.split(scale_maps,   [bs, bs])
        data['lmain']['opacity_maps'], data['rmain']['opacity_maps'] = torch.split(opacity_maps, [bs, bs])

        return data
