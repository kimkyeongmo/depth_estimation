import torch
from torch import nn
from core.extractor import UnetExtractor
from lib.gs_parm_network import GSRegresser
from lib.utils import depth2pc
import sys
import os

sys.path.append('../Video-Depth-Anything')
from video_depth_anything.video_depth import VideoDepthAnything

class VDAGaussianModel(nn.Module):
    def __init__(self, cfg, with_gs_render=True):
        super().__init__()
        self.cfg            = cfg
        self.with_gs_render = with_gs_render

        self.unet_extractor = UnetExtractor(in_channel=3, encoder_dim=self.cfg.raft.encoder_dims)

        self.vda_model = VideoDepthAnything(
            encoder='vits', features=64, out_channels=[48, 96, 192, 384], metric=True
        )
        
        vda_ckpt_path = 'checkpoints/base/vda_120cm_finetuned_epoch3.pth' 
        if os.path.exists(vda_ckpt_path):
            vda_ckpt = torch.load(vda_ckpt_path, map_location='cuda', weights_only=False)
            self.vda_model.load_state_dict(vda_ckpt.get('model', vda_ckpt), strict=True)
            print(f"Loaded VDA from {vda_ckpt_path}")
        else:
            print(f"Warning: VDA ckpt not found at {vda_ckpt_path}")

        if self.with_gs_render:
            self.gs_parm_regresser = GSRegresser(self.cfg, rgb_dim=3, depth_dim=1)

    def forward(self, data, is_train=True):
        # 💡 기준 시점인 0번 뷰(view_0)를 입력으로 사용
        bs        = data['view_0']['img'].shape[0]
        img_input = data['view_0']['img']
        img_gps   = img_input * 2.0 - 1.0

        with torch.cuda.amp.autocast(enabled=True): 
            img_feat = self.unet_extractor(img_gps)

        self.vda_model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                depth_seq, _ = self.vda_model(img_input.unsqueeze(1))
        
        depth_pred = depth_seq.squeeze(1).unsqueeze(1).float()

        assert depth_pred.shape[-2:] == (504, 504), f"Error: VDA Output Resolution Mismatch!"

        # 🚀 병목 제거: 연산량이 큰 LSE(최소제곱법) 정렬을 완전히 제거하고 순수 VDA 출력을 사용
        data['view_0']['depth'] = depth_pred

        if not self.with_gs_render:
            return data, torch.tensor(0.0).to(img_input.device), {}

        data['view_0']['xyz'] = depth2pc(
            depth_pred, data['view_0']['extr'], data['view_0']['intr']
        ).view(bs, -1, 3)
        data['view_0']['pts_valid'] = (depth_pred > 0.1).view(bs, -1)

        rot, scale, opacity = self.gs_parm_regresser(img_gps, depth_pred, img_feat)

        data['view_0']['rot_maps']     = rot.view(bs, 4, img_input.shape[2], img_input.shape[3])
        data['view_0']['scale_maps']   = scale.view(bs, 3, img_input.shape[2], img_input.shape[3])
        data['view_0']['opacity_maps'] = opacity.view(bs, 1, img_input.shape[2], img_input.shape[3])

        return data, torch.tensor(0.0).to(img_input.device), {}