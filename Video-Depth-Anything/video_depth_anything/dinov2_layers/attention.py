# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

from torch import Tensor
from torch import nn
import torch.nn.functional as F


logger = logging.getLogger("dinov2")

# cuda, pytorch rtx 5080에서는 xforemers 사용안함
# try:
#     from xformers.ops import memory_efficient_attention, unbind, fmha

#     XFORMERS_AVAILABLE = True
# except ImportError:
#     #logger.warning("xFormers not available")
#     XFORMERS_AVAILABLE = False
XFORMERS_AVAILABLE = False

class Attention(nn.Module):
    # def __init__(
    #     self,
    #     dim: int,
    #     num_heads: int = 8,
    #     qkv_bias: bool = False,
    #     proj_bias: bool = True,
    #     attn_drop: float = 0.0,
    #     proj_drop: float = 0.0,
    # ) -> None:
    #     super().__init__()
    #     self.num_heads = num_heads
    #     head_dim = dim // num_heads
    #     self.scale = head_dim**-0.5

    #     self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    #     self.attn_drop = nn.Dropout(attn_drop)
    #     self.proj = nn.Linear(dim, dim, bias=proj_bias)
    #     self.proj_drop = nn.Dropout(proj_drop)
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        # SDPA가 자동으로 스케일링을 수행하므로 self.scale 변수는 더 이상 사용하지 않습니다.

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    # def forward(self, x: Tensor) -> Tensor:
    #     B, N, C = x.shape
    #     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

    #     q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    #     attn = q @ k.transpose(-2, -1)

    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)

    #     x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x
    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        # qkv 형태 변환: [B, N, 3, heads, head_dim] -> [3, B, heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # 💡 [핵심] 수동 스케일링(* self.scale) 제거!
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 💡 [핵심] PyTorch 2.0 네이티브 SDPA 적용 (Blackwell 하드웨어 가속)
        x = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# class MemEffAttention(Attention):
#     def forward(self, x: Tensor, attn_bias=None) -> Tensor:
#         if not XFORMERS_AVAILABLE:
#             assert attn_bias is None, "xFormers is required for nested tensors usage"
#             return super().forward(x)
class MemEffAttention(Attention):
    # 이제 MemEffAttention도 억지로 xformers를 부르지 않고 상위 클래스의 SDPA를 그대로 사용합니다.
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if attn_bias is not None:
            raise NotImplementedError("PyTorch SDPA에서는 현재 형태의 attn_bias를 직접 지원하지 않도록 리팩토링 되었습니다.")
        return super().forward(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        