import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import os
import sys
import argparse

from video_depth_anything.video_depth import VideoDepthAnything
from loss.train_loss import VideoDepthLoss 
from dataset import MultiViewVideoDataset

def main():
    if not torch.cuda.is_available():
        print("[Error] CUDA(GPU)를 인식하지 못했습니다! 스크립트를 즉시 종료합니다.")
        sys.exit(1) 
    
    # ==========================================================
    # 💡 [핵심] 거리(80/120)와 초기 가중치를 유연하게 받도록 수정
    # ==========================================================
    parser = argparse.ArgumentParser(description="VDA 80cm/120cm 파인튜닝 통합 스크립트")
    parser.add_argument('--distance', type=int, required=True, choices=[80, 120], 
                        help="학습할 데이터셋의 거리 (80 또는 120)")
    parser.add_argument('--ckpt', type=str, default="checkpoints/metric_video_depth_anything_vits.pth", 
                        help="학습을 시작할 가중치 파일 경로 (기본값: 순정 모델)")
    parser.add_argument('--epochs', type=int, default=5, 
                        help="총 학습 에포크 수 (기본값: 5)")
    args = parser.parse_args()

    # 동적 경로 설정
    DISTANCE = args.distance
    START_CKPT = args.ckpt
    TOTAL_EPOCHS = args.epochs
    
    train_root = f"../thuman_{DISTANCE}cm_render_data/train"
    save_folder = f"checkpoints/{DISTANCE}cm_experiments"
    save_prefix = f"vda_{DISTANCE}cm_finetuned"

    device = torch.device("cuda")
    print(f"\n" + "="*55)
    print(f"🚀 Training on: {torch.cuda.get_device_name(0)}")
    print(f"🎯 Target Distance : {DISTANCE} cm")
    print(f"📂 Dataset Path  : {train_root}")
    print(f"🧠 Start Weights : {START_CKPT}")
    print(f"💾 Save Path     : {save_folder}")
    print("="*55 + "\n")

    if not os.path.exists(train_root):
        print(f"[Error] 데이터셋 경로를 찾을 수 없습니다: {train_root}")
        sys.exit(1)

    torch.set_float32_matmul_precision('high')
    cudnn.benchmark = True

    # 1. 모델 초기화
    model = VideoDepthAnything(
        encoder='vits', features=64, out_channels=[48, 96, 192, 384], metric=True
    ).to(device)

    # 2. 가중치 안전 로드
    if os.path.exists(START_CKPT):
        state_dict = torch.load(START_CKPT, map_location=device)
        # 이전 학습에서 묻은 _orig_mod. 컴파일 접두사 자동 제거
        clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=False)
        print("[Info] 초기 가중치 로드 완료!")
    else:
        print(f"[Error] 가중치 파일을 찾을 수 없습니다: {START_CKPT}")
        sys.exit(1)

    # Encoder(ViT 백본) 동결
    for param in model.pretrained.parameters():
        param.requires_grad = False

    print("[Info] 모델 컴파일 중... (잠시만 기다려주세요)")
    model = torch.compile(model)

    # 3. 데이터 로더 세팅
    train_dataset = MultiViewVideoDataset(train_root, clip_size=4) 
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

    # 4. Loss 및 옵티마이저
    criterion = VideoDepthLoss(reduction="batch-based", is_metric=True).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    scaler = GradScaler('cuda')

    # 5. 학습 루프
    model.train()
    
    for epoch in range(TOTAL_EPOCHS):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{DISTANCE}cm] Epoch {epoch+1}/{TOTAL_EPOCHS}")
        
        for imgs, depths, masks in pbar:
            imgs = imgs.to(device)
            depths = depths.to(device)
            masks = masks.to(device) > 0.5

            optimizer.zero_grad()

            with autocast('cuda'):
                preds = model(imgs)
                
                B, T, H, W = depths.shape
                preds = F.interpolate(preds.flatten(0, 1).unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)
                preds = preds.squeeze(1).unflatten(0, (B, T))

                loss_dict = criterion(prediction=preds, target=depths, mask=masks)
                loss = loss_dict['total_loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 체크포인트 저장 (Epoch 단위)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"{save_prefix}_epoch{epoch+1}.pth")
        
        # 저장할 때는 compile 접두사를 떼고 순수 가중치만 저장
        raw_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()}
        torch.save(raw_state_dict, save_path)
        
    print(f"\n✅ {DISTANCE}cm 데이터셋 총 {TOTAL_EPOCHS} Epoch 학습 완료! 가중치가 {save_folder} 에 저장되었습니다.\n")

if __name__ == "__main__":
    main()