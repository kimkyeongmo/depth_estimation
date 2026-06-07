import torch
import os
import warnings
import math
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def safe_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    scale_factor = scale if scale is not None else (1.0 / math.sqrt(query.size(-1)))
    attn = torch.matmul(query * scale_factor, key.transpose(-2, -1))
    
    if is_causal:
        L = query.size(-2)
        causal_mask = torch.triu(torch.ones(L, L, device=query.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, -10000.0)
        
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, -10000.0)
        else:
            attn_mask = torch.clamp(attn_mask, min=-10000.0)
            attn = attn + attn_mask
            
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, value)
    return out

F.scaled_dot_product_attention = safe_sdpa

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
    CKPT_PATH = "experiments/VDA_GPS_0529_Finetune/ckpt/VDA_GPS_0529_Finetune_final.pth" 
    OUT_DIR = "onnx_models"
    export_to_onnx(CKPT_PATH, OUT_DIR)