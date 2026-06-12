from __future__ import print_function, division

import argparse
import logging
import numpy as np
import cv2
import os
import sys
sys.path.append("./pybind (CudaRuntime.so file)")
import CudaRuntime1

from pathlib import Path
from tqdm import tqdm
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import shaders

from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.utils import get_novel_calib
from lib.GaussianRender import pts2render

import torch
import copy
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class StereoHumanRender:
    def __init__(self, cfg_file, phase):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size
        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)
        self.dataset = StereoHumanDataset(self.cfg.dataset, phase=phase)
        # self.shader_program = self.load_shader(r'.\lib\distortion.vert', r'.\lib\distortion.frag')
        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.eval()
   
   #Stereo/HMD Rendering Pipeline
    def infer_seqence(self, view_select, ratio=0.5):
        CudaRuntime1.init_window(2048, 1028)
#        ipd = 6.05
        total_frames = len(os.listdir(os.path.join(self.cfg.dataset.test_data_root, 'img')))
        for idx in tqdm(range(total_frames)):
            item = self.dataset.get_test_item(idx, source_id=view_select)
            data = self.fetch_data(item)
            data, proj_mat = get_novel_calib(data, self.cfg.dataset, ratio=ratio, intr_key='intr_ori', extr_key='extr_ori')
            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)
                orig_view = data['novel_view']['world_view_transform'][0].clone()
                #orig_full = data['novel_view']['full_proj_transform'][0].clone()
                test_ipd = 0.6
                shift_l = torch.eye(4, device='cuda')
                shift_l[0, 3] = -(test_ipd / 2.0)
        
                shift_r = torch.eye(4, device='cuda')
                shift_r[0, 3] = (test_ipd / 2.0)
                data_left = copy.deepcopy(data)
                data_left['novel_view']['world_view_transform'][0, 3, 0] -= (test_ipd / 2.0)
                data_left['novel_view']['full_proj_transform'] = torch.bmm(data_left['novel_view']['world_view_transform'], proj_mat.cuda().unsqueeze(0))
                data_left['novel_view']['camera_center'] = data_left['novel_view']['world_view_transform'].inverse()[0,3, :3].unsqueeze(0)

                data_right = copy.deepcopy(data)
                data_right['novel_view']['world_view_transform'][0, 3, 0] += (test_ipd / 2.0)
                data_right['novel_view']['full_proj_transform'] = torch.bmm(data_right['novel_view']['world_view_transform'], proj_mat.cuda().unsqueeze(0))
                data_right['novel_view']['camera_center'] = data_right['novel_view']['world_view_transform'].inverse()[0,3, :3].unsqueeze(0)
                output_left, output_right = pts2render(data = data_left,data_r = data_right, bg_color=self.cfg.dataset.bg_color)
                # print(f"{data_left['novel_view']['world_view_transform'][0, 3, 0]}")
                # print(f"{data_right['novel_view']['world_view_transform'][0, 3, 0]}")
            #Real time display
            #self.gl_display(output_left['novel_view']['img_pred'], output_right['novel_view']['img_pred'])
            
            left_img = output_left['novel_view']['img_pred']
            left_img = torch.nn.functional.interpolate(left_img, scale_factor=0.5, mode='bilinear', align_corners=False)
            left_img = left_img[0].permute(1,2,0).contiguous()

            alpha = torch.ones(*left_img.shape[:2], 1, dtype=torch.float32, device=left_img.device)
            left_img_rgba = torch.cat([left_img, alpha], dim=2).contiguous()

            right_img = output_right['novel_view']['img_pred']
            right_img = torch.nn.functional.interpolate(right_img, scale_factor=0.5, mode='bilinear', align_corners=False)
            right_img = right_img[0].permute(1,2,0).contiguous()

            alpha = torch.ones(*right_img.shape[:2], 1, dtype=torch.float32, device=right_img.device)
            right_img_rgba = torch.cat([right_img, alpha], dim=2).contiguous()
            
            stereo = torch.cat([right_img_rgba, left_img_rgba], dim=1).contiguous()

            CudaRuntime1.show_tensor(stereo)
            
                #output_right = pts2render(data_right, bg_color=self.cfg.dataset.bg_color)
                #data = pts2render(data, bg_color=self.cfg.dataset.bg_color)
            
            #GPU memory to CPU memory(output for img)
            #render_origin = self.tensor2np(output['novel_view']['img_pred']).copy()
            
            # render_l = self.tensor2np(output_left['novel_view']['img_pred']).copy()
            # render_r = self.tensor2np(output_right['novel_view']['img_pred']).copy()
            # h, w, _ = render_l.shape
            # cv2.rectangle(render_l, (0, 0), (w, h), (0, 0, 255), 10)
            # cv2.rectangle(render_r, (0, 0), (w, h), (255, 0, 0), 10)
            # render_novel = np.concatenate([render_l, render_r], axis = 1)
            # diff = torch.abs(output_left['novel_view']['img_pred'].float() - output_right['novel_view']['img_pred'].float())
            # print(f"Max Diff: {diff.max().item()}")
            #render_novel = self.tensor2np(data['novel_view']['img_pred'])
            #cv2.imwrite(self.cfg.test_out_path + '/%s_novel.jpg' % (data['name']), render_novel)

        
    #gpu memory to cpu memroy
    def tensor2np(self, img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        img_np = img_np * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np

    def fetch_data(self, data):
        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda().unsqueeze(0)
        return data

    def load_ckpt(self, load_path):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=True)
        logging.info(f"Parameter loading done")
    


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--src_view', type=int, nargs='+', required=True)
    parser.add_argument('--ratio', type=float, default=0.5)
    arg = parser.parse_args()

    cfg = config()
    cfg_for_train = os.path.join('./config', 'stage2.yaml')
    cfg.load(cfg_for_train)
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.batch_size = 1
    cfg.dataset.test_data_root = arg.test_data_root
    cfg.dataset.use_processed_data = False
    cfg.restore_ckpt = arg.ckpt_path
    cfg.test_out_path = './test_out'
    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)
    cfg.freeze()

    render = StereoHumanRender(cfg, phase='test')
    render.infer_seqence(view_select=arg.src_view, ratio=arg.ratio)
