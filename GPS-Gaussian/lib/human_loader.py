import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import logging
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
cv2.setNumThreads(0)

class StereoHumanDataset(Dataset):
    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.phase = phase
        self.data_root = os.path.join(opt.data_root, 'train' if phase == 'train' else 'val')
        
        self.img_path = os.path.join(self.data_root, 'img/%s/%d.jpg')
        self.mask_path = os.path.join(self.data_root, 'mask/%s/%d.png')
        self.depth_path = os.path.join(self.data_root, 'depth/%s/%d.png')
        self.intr_path = os.path.join(self.data_root, 'parm/%s/%d_intrinsic.npy')
        self.extr_path = os.path.join(self.data_root, 'parm/%s/%d_extrinsic.npy')
        self.sample_list = sorted(list(os.listdir(os.path.join(self.data_root, 'img'))))

        cache_name = f'mono_uniform_{opt.src_res}'
        self.local_data_root = os.path.join(opt.data_root, cache_name, self.phase)
        
        if opt.use_processed_data and os.path.exists(self.local_data_root):
            logging.info(f"Using uniformly processed data in {self.local_data_root}")
        elif opt.use_processed_data:
            self.save_local_mono_data()

    def save_local_mono_data(self):
        logging.info(f"Generating uniform resolution ({self.opt.src_res}) data to {self.local_data_root}...")
        
        all_cam_ids = [0, 1, 2, 3, 4] 
        target_res = self.opt.src_res

        for sample_name in self.sample_list:
            for sub in ['/img/', '/mask/', '/parm/', '/depth/']:
                Path(self.local_data_root + sub + sample_name).mkdir(parents=True, exist_ok=True)

            for cam_id in all_cam_ids:
                img = np.array(Image.open(self.img_path % (sample_name, cam_id)))
                mask = np.array(Image.open(self.mask_path % (sample_name, cam_id)))
                depth = cv2.imread(self.depth_path % (sample_name, cam_id), cv2.IMREAD_ANYDEPTH)
                intr = np.load(self.intr_path % (sample_name, cam_id))
                extr = np.load(self.extr_path % (sample_name, cam_id))

                H, W = img.shape[:2]
                S = max(H, W)
                padded_img = np.zeros((S, S, 3), dtype=np.uint8)
                padded_mask = np.zeros((S, S), dtype=np.uint8)
                padded_depth = np.zeros((S, S), dtype=depth.dtype)
                
                pad_t, pad_l = (S - H) // 2, (S - W) // 2
                padded_img[pad_t:pad_t+H, pad_l:pad_l+W] = img
                padded_mask[pad_t:pad_t+H, pad_l:pad_l+W] = mask[:,:,0] if mask.ndim==3 else mask
                padded_depth[pad_t:pad_t+H, pad_l:pad_l+W] = depth

                final_img = cv2.resize(padded_img, (target_res, target_res), interpolation=cv2.INTER_AREA)
                final_mask = cv2.resize(padded_mask, (target_res, target_res), interpolation=cv2.INTER_NEAREST)
                final_depth = cv2.resize(padded_depth, (target_res, target_res), interpolation=cv2.INTER_NEAREST)

                scale = target_res / float(S)
                intr[0, 2] = (intr[0, 2] + pad_l) * scale 
                intr[1, 2] = (intr[1, 2] + pad_t) * scale 
                intr[0, 0] *= scale                        
                intr[1, 1] *= scale                        

                cv2.imwrite(os.path.join(self.local_data_root, f'img/{sample_name}/{cam_id}.jpg'), final_img[:,:,::-1])
                cv2.imwrite(os.path.join(self.local_data_root, f'mask/{sample_name}/{cam_id}.png'), final_mask)
                cv2.imwrite(os.path.join(self.local_data_root, f'depth/{sample_name}/{cam_id}.png'), final_depth)
                np.save(os.path.join(self.local_data_root, f'parm/{sample_name}/{cam_id}_intr.npy'), intr)
                np.save(os.path.join(self.local_data_root, f'parm/{sample_name}/{cam_id}_extr.npy'), extr)

    def get_novel_view_tensor(self, sample_name, view_id):
        img = cv2.imread(os.path.join(self.local_data_root, f'img/{sample_name}/{view_id}.jpg'))
        mask = cv2.imread(os.path.join(self.local_data_root, f'mask/{sample_name}/{view_id}.png'), 0)
        depth = cv2.imread(os.path.join(self.local_data_root, f'depth/{sample_name}/{view_id}.png'), cv2.IMREAD_ANYDEPTH)
        intr = np.load(os.path.join(self.local_data_root, f'parm/{sample_name}/{view_id}_intr.npy'))
        extr = np.load(os.path.join(self.local_data_root, f'parm/{sample_name}/{view_id}_extr.npy'))
        
        img_t = torch.from_numpy(img[:,:,::-1].copy()).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0) / (2.0 ** 15)

        mask_t[mask_t < 0.5] = 0.0
        mask_t[mask_t >= 0.5] = 1.0
        
        img_t = img_t * mask_t
        depth_t = depth_t * mask_t
        h, w = img.shape[:2] 

        R, T = extr[:3, :3].T, extr[:3, 3]
        
        projection_matrix = getProjectionMatrix(znear=self.opt.znear, zfar=self.opt.zfar, K=intr, h=h, w=w).transpose(0, 1)
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.opt.trans), self.opt.scale)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        
        return {
            'img': img_t, 
            'mask': mask_t,
            'depth_gt': depth_t,
            'intr': torch.FloatTensor(intr), 'extr': torch.FloatTensor(extr),
            'width': w, 'height': h, 'FovX': focal2fov(intr[0,0], w), 'FovY': focal2fov(intr[1,1], h),
            'world_view_transform': world_view_transform, 'full_proj_transform': full_proj_transform,
            'camera_center': world_view_transform.inverse()[3, :3]
        }
    
    def __getitem__(self, index):
        sample_name = self.sample_list[index % len(self.sample_list)]
        data = {'name': sample_name}
        
        for view_idx in [0, 1, 2, 3, 4]:
            data[f'view_{view_idx}'] = self.get_novel_view_tensor(sample_name, view_idx)
            
        return data

    def __len__(self):
        return len(self.sample_list) * (50 if self.phase == 'train' else 1)