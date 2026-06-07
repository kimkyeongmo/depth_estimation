import os
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from video_depth_anything.video_depth import VideoDepthAnything
from benchmark.eval import metric  

# ==============================================================================
# [수정 1] meshgrid 경고 제거 및 타입 동기화 추가
# ==============================================================================
def compute_errors_torch(gt, pred):
    return torch.mean(torch.abs(gt - pred) / gt)

def tae_torch(depth1, depth2, R_2_1, T_2_1, K, mask):
    H, W = depth1.shape
    
    K = K.to(dtype=depth1.dtype)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij')
    xx, yy = xx.t(), yy.t()  

    xx = xx.to(dtype=depth1.dtype, device=depth1.device)
    yy = yy.to(dtype=depth1.dtype, device=depth1.device)
    
    X = (xx - cx) * depth1 / fx
    Y = (yy - cy) * depth1 / fy
    Z = depth1
    points3d = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1) 

    R_2_1 = R_2_1.to(dtype=depth1.dtype)
    T_2_1 = T_2_1.to(dtype=depth1.dtype)

    points3d_transformed = torch.matmul(points3d, R_2_1.T) + T_2_1
    X_world, Y_world, Z_world = points3d_transformed[:, 0], points3d_transformed[:, 1], points3d_transformed[:, 2]
    
    X_plane = (X_world * fx) / Z_world + cx
    Y_plane = (Y_world * fy) / Z_world + cy

    X_plane = torch.round(X_plane).to(dtype=torch.long)
    Y_plane = torch.round(Y_plane).to(dtype=torch.long)

    valid_mask = (X_plane >= 0) & (X_plane < W) & (Y_plane >= 0) & (Y_plane < H)
    if valid_mask.sum() == 0:
        return 0.0

    depth_proj = torch.zeros((H, W), dtype=depth1.dtype, device=depth1.device)

    valid_X = X_plane[valid_mask]
    valid_Y = Y_plane[valid_mask]
    valid_Z = Z_world[valid_mask]

    depth_proj[valid_Y, valid_X] = valid_Z

    valid_mask = (depth_proj > 0) & (depth2 > 0) & mask
    if valid_mask.sum() == 0:
        return 0.0
        
    abs_errors = compute_errors_torch(depth2[valid_mask], depth_proj[valid_mask])
    return abs_errors.item()

def calculate_clip_tae(preds, gts, masks, Ks, Es):
    error_sum = 0.0
    valid_pairs = 0
    
    for i in range(len(preds) - 1):
        depth1 = preds[i]
        depth2 = preds[i+1]
        mask1 = masks[i]
        mask2 = masks[i+1]
        K = Ks[i] 
        
        T_1 = Es[i] 
        T_2 = Es[i+1]
        
        T_2_1 = torch.inverse(T_2) @ T_1
        R_2_1 = T_2_1[:3, :3]
        t_2_1 = T_2_1[:3, 3]
        
        error1 = tae_torch(depth1, depth2, R_2_1, t_2_1, K, mask2)
        
        T_1_2 = torch.inverse(T_2_1)
        R_1_2 = T_1_2[:3, :3]
        t_1_2 = T_1_2[:3, 3]
        
        error2 = tae_torch(depth2, depth1, R_1_2, t_1_2, K, mask1)
        
        error_sum += (error1 + error2)
        valid_pairs += 2
        
    if valid_pairs == 0: return 0.0
    return (error_sum / valid_pairs) * 100.0 

# ==============================================================================

class THumanValDataset(Dataset):
    def __init__(self, root_dir, clip_size=4, target_size=(518, 518)):
        self.root_dir = root_dir
        self.clip_size = clip_size
        self.target_size = target_size
        self.img_base = os.path.join(root_dir, 'img')
        self.depth_base = os.path.join(root_dir, 'depth')
        self.mask_base = os.path.join(root_dir, 'mask')
        self.parm_base = os.path.join(root_dir, 'parm')

        folders = sorted([f for f in os.listdir(self.depth_base) if os.path.isdir(os.path.join(self.depth_base, f))])
        self.clips = []
        for f_name in folders:
            frames = sorted([f for f in os.listdir(os.path.join(self.depth_base, f_name)) if f.endswith('.png')])
            for i in range(len(frames) - clip_size + 1):
                c_data = []
                for f in frames[i : i + clip_size]:
                    b = os.path.splitext(f)[0]
                    i_p = os.path.join(self.img_base, f_name, b + '_hr.jpg')
                    if not os.path.exists(i_p): 
                        i_p = os.path.join(self.img_base, f_name, b + '.jpg')
                        
                    c_data.append({
                        'img': i_p, 'depth': os.path.join(self.depth_base, f_name, f),
                        'mask': os.path.join(self.mask_base, f_name, f),
                        'k': os.path.join(self.parm_base, f_name, f'{b}_intrinsic.npy'),
                        'e': os.path.join(self.parm_base, f_name, f'{b}_extrinsic.npy')
                    })
                self.clips.append(c_data)

    def __len__(self): 
        return len(self.clips)

    def __getitem__(self, idx):
        paths = self.clips[idx]
        imgs, dpts, msks, ks, es = [], [], [], [], []
        H_t, W_t = self.target_size

        for p in paths:
            raw_img = cv2.imread(p['img'])
            orig_H, orig_W = raw_img.shape[:2]
            img = cv2.resize(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), (W_t, H_t))
            imgs.append(torch.tensor(img.transpose(2, 0, 1) / 255.0, dtype=torch.float32))

            d = cv2.imread(p['depth'], cv2.IMREAD_ANYDEPTH).astype(np.float32) / 32768.0
            dpts.append(torch.tensor(cv2.resize(d, (W_t, H_t), interpolation=cv2.INTER_NEAREST)))

            m = cv2.imread(p['mask'], cv2.IMREAD_GRAYSCALE)
            m = (m > 127).astype(np.float32) if m is not None else np.ones_like(d)
            msks.append(torch.tensor(cv2.resize(m, (W_t, H_t), interpolation=cv2.INTER_NEAREST)))

            K = np.load(p['k'])
            K[0, :] *= (W_t / float(orig_W))
            K[1, :] *= (H_t / float(orig_H))
            ks.append(torch.tensor(K, dtype=torch.float32))

            E = np.eye(4, dtype=np.float32)
            E[:3, :4] = np.load(p['e'])
            es.append(torch.tensor(E))

        return torch.stack(imgs), torch.stack(dpts), torch.stack(msks), torch.stack(ks), torch.stack(es)

def main():
    if not torch.cuda.is_available():
        print("[Error] CUDA(GPU)를 인식하지 못했습니다! 스크립트를 즉시 종료합니다.")
        sys.exit(1)
        
    device = torch.device("cuda")
    print(f"Evaluating on: {torch.cuda.get_device_name(0)}")

    # 모델 인스턴스는 한 번만 생성하고 컴파일 (가중치만 계속 갈아끼움)
    model = VideoDepthAnything(encoder='vits', features=64, out_channels=[48, 96, 192, 384], metric=True).to(device)
    model = torch.compile(model)
    
    # 💡 [핵심 수정] 각 모델 그룹에 맞는 검증 데이터(val_data) 경로를 추가로 매핑합니다.
    experiments = [
        {
            "group": "80cm_Model", 
            "dir": "checkpoints/80cm_experiments", 
            "prefix": "vda_80cm_finetuned_epoch",
            "val_data": "../../thuman_80cm_render_data/val"  # 80cm 전용 검증셋
        },
        {
            "group": "120cm_Model", 
            "dir": "checkpoints/120cm_experiments", 
            "prefix": "vda_120cm_finetuned_epoch",
            "val_data": "../../thuman_120cm_render_data/val" # 120cm 전용 검증셋
        }
    ]
    epochs_to_evaluate = [1, 2, 3, 4, 5]
    
    summary_results = []

    # 1. 모델 그룹 단위 루프
    for exp in experiments:
        
        # 💡 [핵심 수정] 평가 그룹에 맞춰 데이터셋을 동적으로 로드합니다.
        val_data_path = exp['val_data']
        if not os.path.exists(val_data_path):
            print(f"\n[Error] {exp['group']} 평가를 위한 검증 데이터 경로를 찾을 수 없습니다: {val_data_path}")
            print("해당 모델 그룹의 평가를 건너뜁니다...\n")
            continue
            
        print("\n" + "*"*80)
        print(f" 📦 Loading Validation Dataset for [{exp['group']}] ... ")
        print(f" 📂 Path: {val_data_path}")
        print("*"*80)
        
        val_loader = DataLoader(THumanValDataset(val_data_path), batch_size=16, num_workers=8, pin_memory=True)

        # 2. 에포크 단위 루프
        for epoch in epochs_to_evaluate:
            ckpt_name = f"{exp['prefix']}{epoch}.pth"
            ckpt_path = os.path.join(exp['dir'], ckpt_name)
            
            if not os.path.exists(ckpt_path):
                print(f"[Warning] Checkpoint not found: {ckpt_path}. Skipping...")
                continue
                
            print("\n" + "="*80)
            print(f" 🚀 Evaluating [{exp['group']}] : Epoch {epoch} ")
            print("="*80)

            state_dict = torch.load(ckpt_path, map_location=device)
            
            compiled_state_dict = {}
            for k, v in state_dict.items():
                compiled_state_dict[f"_orig_mod.{k}"] = v
                
            model.load_state_dict(compiled_state_dict)
            model.eval()

            total_abs_rel = 0.0
            total_rmse = 0.0
            total_tae = 0.0
            num_frames = 0
            num_clips = 0

            pbar = tqdm(val_loader, desc=f"Progress")
            
            with torch.no_grad():
                for imgs, depths, masks, Ks, Es in pbar:
                    imgs = imgs.to(device)
                    depths = depths.to(device)
                    masks = masks.to(device) > 0.5 
                    Ks = Ks.to(device)
                    Es = Es.to(device)

                    B, T, C, H, W = imgs.shape

                    with torch.amp.autocast('cuda'):
                        preds, _ = model(imgs)
                    
                    preds = preds.float()

                    _, _, eval_H, eval_W = depths.shape
                    preds = F.interpolate(preds.flatten(0, 1).unsqueeze(1), size=(eval_H, eval_W), mode="bilinear", align_corners=False)
                    preds = preds.squeeze(1).unflatten(0, (B, T))

                    for b in range(B):
                        clip_pred = preds[b]
                        clip_gt = depths[b]
                        clip_mask = masks[b]
                        
                        valid_clip_mask = clip_mask & (clip_gt > 1e-3)
                        
                        if valid_clip_mask.sum() > 0:
                            gt_median = torch.median(clip_gt[valid_clip_mask])
                            pred_median = torch.median(clip_pred[valid_clip_mask])
                            
                            if pred_median > 1e-5:
                                scale_factor = gt_median / pred_median
                                clip_pred = clip_pred * scale_factor

                        for t in range(T):
                            _p = clip_pred[t].unsqueeze(0)
                            _g = clip_gt[t].unsqueeze(0)
                            _m = clip_mask[t].unsqueeze(0) & (_g > 1e-3) 
                            
                            if _m.sum() > 0:
                                abs_rel = metric.abs_relative_difference(_p, _g, valid_mask=_m).item()
                                rmse = metric.rmse_linear(_p, _g, valid_mask=_m).item()
                                
                                total_abs_rel += abs_rel
                                total_rmse += rmse
                                num_frames += 1

                        clip_K = Ks[b]
                        clip_E = Es[b]
                        tae = calculate_clip_tae(clip_pred, clip_gt, clip_mask, clip_K, clip_E)
                        
                        total_tae += tae
                        num_clips += 1

                    current_abs_rel = total_abs_rel / max(1, num_frames)
                    current_rmse = total_rmse / max(1, num_frames)
                    current_tae = total_tae / max(1, num_clips)
                    
                    pbar.set_postfix({
                        'AbsRel': f"{current_abs_rel:.3f}", 
                        'RMSE': f"{current_rmse:.3f}", 
                        'TAE': f"{current_tae:.3f}"
                    })

            mean_abs_rel = total_abs_rel / max(1, num_frames)
            mean_rmse = total_rmse / max(1, num_frames)
            mean_tae = total_tae / max(1, num_clips)
            
            summary_results.append({
                'experiment': exp['group'],
                'epoch': epoch,
                'abs_rel': mean_abs_rel,
                'rmse': mean_rmse,
                'tae': mean_tae
            })
            
            print(f" -> Results: AbsRel: {mean_abs_rel:.4f} | RMSE: {mean_rmse:.4f} | TAE: {mean_tae:.4f} %\n")

    print("\n" + "="*80)
    print(" [Evaluation Summary: 80cm vs 120cm Models on Target Domains] ".center(80))
    print("="*80)
    print(f"{'Model Group':<20} | {'Epoch':<6} | {'Abs Rel':<10} | {'RMSE':<10} | {'TAE (%)':<10}")
    print("-" * 80)
    
    current_exp = ""
    for res in summary_results:
        if current_exp != "" and current_exp != res['experiment']:
            print("-" * 80)
        current_exp = res['experiment']
        
        print(f"{res['experiment']:<20} | {res['epoch']:<6} | {res['abs_rel']:<10.4f} | {res['rmse']:<10.4f} | {res['tae']:<10.4f}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()