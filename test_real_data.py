from __future__ import print_function, division

import argparse
import logging
import numpy as np
import cv2
import os
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
        self.shader_program = self.load_shader(r'.\lib\distortion.vert', r'.\lib\distortion.frag')
        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.eval()
   
   #Stereo/HMD Rendering Pipeline
    def infer_seqence(self, view_select, ratio=0.5):
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
            self.gl_display(output_left['novel_view']['img_pred'], output_right['novel_view']['img_pred'])

            
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

    def gl_display(self, render_l, render_r):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0,0,960, 1080)
        self.gl_cal(render_l)
        glViewport(960, 0, 960, 1080)
        self.gl_cal(render_r)
        glutSwapBuffers()
        glutMainLoopEvent()
        cv2.waitKey(1)
        
    def gl_cal(self, img_render):
        
        #shader
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        #tex_id에 2D텍스쳐 작업 할당
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        #해당 변수명에 값 전달
        glUniform1i(glGetUniformLocation(self.shader_program, "screenTexture"), 0)
        glUniform1f(glGetUniformLocation(self.shader_program, "K1"), 1.0)
        glUniform1f(glGetUniformLocation(self.shader_program, "K2"), 1.0)
        
        img_data = img_render.squeeze(0).permute(1,2,0).contiguous()
        torch.cuda.synchronize()

        #GPU memory to CPU memory (testing)
        img_np = img_render.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        H, W, _ = img_np.shape
        #img_data = img_render.squeeze(0).permute(1,2,0).contiguous()
        torch.cuda.synchronize()
        #glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_FLOAT, img_data.data_ptr())
        #texture mapping
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_FLOAT, img_np)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(-1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, 1)
        glTexCoord2f(0, 0); glVertex2f(-1, 1)
        glEnd()
        #shader 0
        glUseProgram(0)
        
    def load_shader(self, vert_path, frag_path):
        with open(vert_path, 'r', encoding='utf-8') as f:
            vert_code = f.read()
        with open(frag_path, 'r', encoding='utf-8') as f:
            frag_code = f.read()
        vs = shaders.compileShader(vert_code, GL_VERTEX_SHADER)
        fs = shaders.compileShader(frag_code, GL_FRAGMENT_SHADER)
        return shaders.compileProgram(vs, fs, validate=False)
        
    
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
    import sys
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(1920, 1080)
    glutCreateWindow(b"Test")
    glutDisplayFunc(lambda: None)
    
    
    glEnable(GL_TEXTURE_2D)
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)


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
    render.tex_id = tex_id
    render.infer_seqence(view_select=arg.src_view, ratio=arg.ratio)
