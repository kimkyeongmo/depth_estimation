import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset

class MultiViewVideoDataset(Dataset):
    def __init__(self, root_dir, clip_size=4):
        self.root_dir = root_dir
        self.clip_size = clip_size
        
        self.img_base = os.path.join(root_dir, 'img')
        self.depth_base = os.path.join(root_dir, 'depth')
        self.mask_base = os.path.join(root_dir, 'mask')

        # data_id_000 ~ data_id_015 형태의 폴더들
        cam_folders = sorted([f for f in os.listdir(self.depth_base) if os.path.isdir(os.path.join(self.depth_base, f))])
        self.clips = []
        
        # 1. 렌더링 스크립트의 물리적 카메라 회전 궤적 정의
        # 0: Left, 2/3/4: Intermediate, 1: Right
        self.physical_order = ['0', '2', '3', '4', '1']
        
        for folder_name in cam_folders:
            img_dir = os.path.join(self.img_base, folder_name)
            depth_dir = os.path.join(self.depth_base, folder_name)
            mask_dir = os.path.join(self.mask_base, folder_name)

            # 폴더 내에 0, 1, 2, 3, 4 에 해당하는 5개 세트가 존재한다고 가정
            # 시간적 순서(물리적 궤적)에 맞게 5장의 파일 경로를 정리
            scene_frames = []
            valid_scene = True
            
            for frame_idx in self.physical_order:
                # _hr.jpg 우선 확인 로직
                img_path = os.path.join(img_dir, frame_idx + '.jpg')
                if not os.path.exists(img_path):
                    img_path = os.path.join(img_dir, frame_idx + '.png')
                    
                depth_path = os.path.join(depth_dir, frame_idx + '.png')
                mask_path = os.path.join(mask_dir, frame_idx + '.png')

                if not os.path.exists(depth_path):
                    valid_scene = False
                    break
                    
                scene_frames.append({
                    'img': img_path,
                    'depth': depth_path,
                    'mask': mask_path
                })
            
            # 폴더 내 5장 파일이 모두 유효할 경우에만 클립 구성 진행
            if valid_scene:
                # 2. 5장의 프레임 중 무작위로 4장(clip_size)을 시간 순서대로 추출 (Time-Skip Augmentation)
                # 만약 clip_size가 4가 아니라 다른 값이라도 유연하게 대응
                if self.clip_size < len(self.physical_order):
                    # 0~4 사이의 인덱스 중 4개를 무작위 추출 후 정렬 (물리적 순서 보장)
                    selected_indices = random.sample(range(len(self.physical_order)), self.clip_size)
                    selected_indices.sort()
                    clip_file_paths = [scene_frames[i] for i in selected_indices]
                else:
                    clip_file_paths = scene_frames[:self.clip_size]
                    
                self.clips.append(clip_file_paths)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_paths = self.clips[idx] 
        clip_images, clip_depths, clip_masks = [], [], []
        new_H, new_W = 518, 518

        for paths in clip_paths:
            img = cv2.imread(paths['img'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Metric 깊이 복원 
            depth = cv2.imread(paths['depth'], cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32) / 32768.0 

            mask = cv2.imread(paths['mask'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32) if mask is not None else np.ones_like(depth)

            # 리사이즈
            img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))

            clip_images.append(torch.tensor(img))
            clip_depths.append(torch.tensor(depth))
            clip_masks.append(torch.tensor(mask))

        return torch.stack(clip_images), torch.stack(clip_depths), torch.stack(clip_masks)