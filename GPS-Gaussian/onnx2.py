import os
import math
import torch
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from lib.network import VDAGaussianModel
from config.stereo_human_config import ConfigStereoHuman as config


# =================================================================
# VDA 8포트 캐시 wrapper (몽키패치 적용 전에 export)
# =================================================================
class VDA_TRT_Explicit_Wrapper(torch.nn.Module):
    def __init__(self, vda):
        super().__init__()
        self.vda = vda

    def forward(self, image, c0, c1, c2, c3, c4, c5, c6, c7):
        feat = self.vda.pretrained.get_intermediate_layers(
            image.flatten(0, 1),
            self.vda.intermediate_layer_idx[self.vda.encoder],
            return_class_token=True
        )
        explicit_cache_list = [c0, c1, c2, c3, c4, c5, c6, c7]
        B, T, C, H, W = image.shape
        patch_h, patch_w = H // 14, W // 14
        depth, feature_map, new_cache = self.vda.head(
            feat, patch_h, patch_w, T,
            cached_hidden_state_list=explicit_cache_list
        )
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return (depth.squeeze(1), feature_map,
                new_cache[0], new_cache[1], new_cache[2], new_cache[3],
                new_cache[4], new_cache[5], new_cache[6], new_cache[7])


# =================================================================
# U-Net/GS 전용 몽키패치 4종 (document 12에서 검증된 정상 조합)
# =================================================================
def apply_monkey_patches():
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
        return torch.matmul(attn, value)
    F.scaled_dot_product_attention = safe_sdpa

    orig_interpolate = F.interpolate
    def safe_interpolate(input, size=None, scale_factor=None, mode='nearest',
                         align_corners=None, recompute_scale_factor=None, antialias=False):
        if mode == 'bicubic':
            mode = 'bilinear'
        return orig_interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor, False)
    F.interpolate = safe_interpolate

    orig_conv2d = F.conv2d
    def safe_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        return orig_conv2d(input, weight, bias, stride, padding, dilation, groups)
    F.conv2d = safe_conv2d

    orig_linear = F.linear
    def safe_linear(input, weight, bias=None):
        if input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        return orig_linear(input, weight, bias)
    F.linear = safe_linear


def export_all(ckpt_path, out_dir, render_res=504):
    os.makedirs(out_dir, exist_ok=True)

    cfg_wrapper = config()
    cfg_wrapper.load("config/stage2.yaml")
    cfg = cfg_wrapper.get_cfg()
    cfg.defrost()
    cfg.dataset.src_res = render_res
    cfg.freeze()

    model = VDAGaussianModel(cfg, with_gs_render=True).cuda().eval()
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['network'], strict=True)
    model = model.to(torch.float32)

    # ============================================================
    # 1. VDA 먼저 export — 몽키패치 적용 전 (VDA depth interpolate 보호)
    # ============================================================
    print("🚀 Exporting VDA (8-Port, FP32, no monkey-patch)...")
    dummy_img = torch.rand((1, 1, 3, render_res, render_res), dtype=torch.float32, device='cuda')
    with torch.no_grad():
        feat = model.vda_model.pretrained.get_intermediate_layers(
            dummy_img.flatten(0, 1),
            model.vda_model.intermediate_layer_idx[model.vda_model.encoder],
            return_class_token=True
        )
        _, _, init_cache = model.vda_model.head(
            feat, render_res // 14, render_res // 14, 1, cached_hidden_state_list=None
        )
    dummy_caches = []
    for c in init_cache:
        shape = list(c.shape); shape[1] = 31
        dummy_caches.append(torch.zeros(shape, dtype=torch.float32, device='cuda'))

    wrapper = VDA_TRT_Explicit_Wrapper(model.vda_model).eval()
    torch.onnx.export(
        wrapper, (dummy_img, *dummy_caches),
        os.path.join(out_dir, "vda_model.onnx"),
        input_names=['input_image'] + [f'in_cache_{i}' for i in range(8)],
        output_names=['depth_out', 'features'] + [f'out_cache_{i}' for i in range(8)],
        opset_version=18, do_constant_folding=True
    )

    # ============================================================
    # 2. 이제 몽키패치 적용 후 U-Net/GS export (정상 조합)
    # ============================================================
    apply_monkey_patches()
    print("🚀 Exporting U-Net Extractor (FP32, monkey-patched)...")
    dummy_img_gps = torch.rand(1, 3, render_res, render_res, dtype=torch.float32, device='cuda')
    torch.onnx.export(
        model.unet_extractor, (dummy_img_gps,),
        os.path.join(out_dir, "unet_extractor.onnx"),
        input_names=['input_image_gps'], output_names=['feat1', 'feat2', 'feat3'],
        opset_version=18
    )

    print("🚀 Exporting GSRegresser (FP32, monkey-patched)...")
    dummy_depth = torch.rand(1, 1, render_res, render_res, dtype=torch.float32, device='cuda')
    dummy_feat = (
        torch.rand(1, cfg.raft.encoder_dims[0], 252, 252, dtype=torch.float32, device='cuda'),
        torch.rand(1, cfg.raft.encoder_dims[1], 126, 126, dtype=torch.float32, device='cuda'),
        torch.rand(1, cfg.raft.encoder_dims[2], 63, 63, dtype=torch.float32, device='cuda')
    )
    torch.onnx.export(
        model.gs_parm_regresser, (dummy_img_gps, dummy_depth, dummy_feat),
        os.path.join(out_dir, "gs_regresser.onnx"),
        input_names=['img_gps', 'depth', 'feat1', 'feat2', 'feat3'],
        output_names=['rot_maps', 'scale_maps', 'opacity_maps'],
        opset_version=18
    )
    print("✅ Done.")


if __name__ == "__main__":
    CKPT_PATH = "checkpoints/VDA_GPS_0529_Finetune_final.pth"
    OUT_DIR = "onnx_models"
    export_all(CKPT_PATH, OUT_DIR)