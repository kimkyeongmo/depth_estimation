import torch
import os
import warnings

warnings.filterwarnings("ignore")

# 💡 [핵심 수정 1] safe_sdpa 몽키패치 완전 삭제
# PyTorch 2.7.x 및 Opset 18 환경에서는 SDPA를 네이티브 노드로 정상 추출합니다.
# 수동으로 수학 연산을 쪼개면 TensorRT의 FlashAttention 가속이 비활성화됩니다.

from lib.network import VDAGaussianModel
from config.stereo_human_config import ConfigStereoHuman as config

def export_to_onnx(ckpt_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    cfg_wrapper = config()
    cfg_wrapper.load("config/stage2.yaml") 
    cfg = cfg_wrapper.get_cfg()
    cfg.defrost()
    cfg.dataset.src_res = 504
    cfg.freeze()

    # 모델 초기화 및 가중치 로드
    model = VDAGaussianModel(cfg, with_gs_render=True).cuda().eval()
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['network'], strict=True)
    
    model = model.to(torch.float32)

    print("Exporting VDA Model (FP32 ONNX)...")
    dummy_img_vda = torch.rand(1, 1, 3, 504, 504, dtype=torch.float32, device='cuda')
    torch.onnx.export(
        model.vda_model, 
        (dummy_img_vda,), 
        os.path.join(out_dir, "vda_model.onnx"),
        input_names=['input_image'], 
        output_names=['depth_out', 'features'],
        opset_version=18
    )

    print("Exporting U-Net Extractor (FP32 ONNX)...")
    dummy_img_gps = torch.rand(1, 3, 504, 504, dtype=torch.float32, device='cuda')
    torch.onnx.export(
        model.unet_extractor, 
        (dummy_img_gps,), 
        os.path.join(out_dir, "unet_extractor.onnx"),
        input_names=['input_image_gps'], 
        output_names=['feat1', 'feat2', 'feat3'],
        opset_version=18
    )

    print("Exporting GSRegresser (FP32 ONNX)...")
    dummy_depth = torch.rand(1, 1, 504, 504, dtype=torch.float32, device='cuda')
    dummy_feat = (
        torch.rand(1, cfg.raft.encoder_dims[0], 252, 252, dtype=torch.float32, device='cuda'),
        torch.rand(1, cfg.raft.encoder_dims[1], 126, 126, dtype=torch.float32, device='cuda'),
        torch.rand(1, cfg.raft.encoder_dims[2], 63, 63, dtype=torch.float32, device='cuda')
    )
    torch.onnx.export(
        model.gs_parm_regresser, 
        (dummy_img_gps, dummy_depth, dummy_feat), 
        os.path.join(out_dir, "gs_regresser.onnx"),
        input_names=['img_gps', 'depth', 'feat1', 'feat2', 'feat3'],
        output_names=['rot_maps', 'scale_maps', 'opacity_maps'],
        opset_version=18
    )

if __name__ == "__main__":
    CKPT_PATH = "checkpoints/VDA_GPS_0529_Finetune_final.pth" 
    OUT_DIR = "onnx_models"
    export_to_onnx(CKPT_PATH, OUT_DIR)