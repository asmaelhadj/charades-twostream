"""Self-contained VideoMAE V2 fine-tuning backbone (ViT) for Charades.

This is a faithful, dependency-light re-implementation of the *fine-tuning*
VisionTransformer from the official repo (OpenGVLab/VideoMAEv2,
`models/modeling_finetune.py`). It is kept self-contained on purpose:

  * Narval compute nodes have NO internet, so we cannot `pip install timm`
    at run time nor clone the repo on the fly.
  * The Alliance wheelhouse may not carry the exact `timm==0.4.12` the repo
    pins, so we vendor the two tiny helpers we need (`drop_path`,
    `trunc_normal_`) instead of depending on it.

The module/parameter names match the official checkpoints exactly, so the
released weights (e.g. `vit_b_k710_dl_from_giant.pth`) load with
`strict=False` after dropping the pretrained classification head.

Input  : video clip tensor [B, C, T, H, W] = [B, 3, 16, 224, 224]
Output : raw logits [B, num_classes]  (apply sigmoid for multi-label probs)
"""
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Vendored helpers (normally from timm) — keep this file dependency-light.
# --------------------------------------------------------------------------- #
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Stochastic depth per sample (applied on the residual branch)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    return x.div(keep_prob) * mask


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """In-place truncated normal init (simplified timm version)."""
    with torch.no_grad():
        l = (1.0 + math.erf((a - mean) / (std * math.sqrt(2.0)))) / 2.0
        u = (1.0 + math.erf((b - mean) / (std * math.sqrt(2.0)))) / 2.0
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# --------------------------------------------------------------------------- #
# Transformer building blocks (names match the official checkpoint).
# --------------------------------------------------------------------------- #
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with separate q/v bias (BEiT/VideoMAE style)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias,
            ))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, init_values=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """3D (tubelet) patch embedding: [B, C, T, H, W] -> [B, N, embed_dim]."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 num_frames=16, tubelet_size=2):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.tubelet_size = tubelet_size
        self.num_patches = (
            (img_size // patch_size) * (img_size // patch_size)
            * (num_frames // tubelet_size)
        )
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        _, _, _, h, w = x.shape
        assert (h, w) == self.img_size, (
            f"Input size ({h}x{w}) != model ({self.img_size[0]}x{self.img_size[1]})"
        )
        return self.proj(x).flatten(2).transpose(1, 2)


def get_sinusoid_encoding_table(n_position, d_hid):
    """Fixed sine-cosine positional embedding [1, n_position, d_hid]."""
    def angle_vec(pos):
        return [pos / np.power(10000, 2 * (j // 2) / d_hid) for j in range(d_hid)]

    table = np.array([angle_vec(p) for p in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.tensor(table, dtype=torch.float32).unsqueeze(0)


class VisionTransformer(nn.Module):
    """VideoMAE V2 fine-tuning ViT (mean-pooling head, no class token)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=157,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 head_drop_rate=0.0, norm_layer=None, init_values=0.0, all_frames=16,
                 tubelet_size=2, use_mean_pooling=True, init_scale=0.001,
                 grad_checkpoint=False):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.grad_checkpoint = grad_checkpoint

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim,
                                      num_frames=all_frames, tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches

        # Fixed (non-learnable) positional table: not a parameter, so it is
        # absent from the checkpoint. Registered as a non-persistent buffer so
        # it follows .cuda()/.to() but stays out of state_dict (matches repo).
        self.register_buffer(
            "pos_embed", get_sinusoid_encoding_table(num_patches, embed_dim),
            persistent=False,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, init_values=init_values)
            for i in range(depth)
        ])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head_dropout = nn.Dropout(head_drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {"pos_embed"}

    def forward_features(self, x):
        b = x.size(0)
        x = self.patch_embed(x)
        x = x + self.pos_embed.expand(b, -1, -1).type_as(x).to(x.device).detach()
        x = self.pos_drop(x)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        return self.norm(x[:, 0])

    def forward(self, x, return_features=False):
        feats = self.forward_features(x)
        logits = self.head(self.head_dropout(feats))
        if return_features:
            return logits, feats
        return logits


# --------------------------------------------------------------------------- #
# Model factory + checkpoint loading.
# --------------------------------------------------------------------------- #
_ARCH = {
    "vit_small_patch16_224": dict(embed_dim=384, depth=12, num_heads=6),
    "vit_base_patch16_224": dict(embed_dim=768, depth=12, num_heads=12),
    "vit_large_patch16_224": dict(embed_dim=1024, depth=24, num_heads=16),
    "vit_huge_patch16_224": dict(embed_dim=1280, depth=32, num_heads=16),
}


def _clean_state_dict(ckpt):
    """Unwrap common checkpoint containers and strip key prefixes."""
    for key in ("module", "model", "model_state", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break
    out = {}
    for k, v in ckpt.items():
        for prefix in ("module.", "backbone.", "encoder."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        out[k] = v
    return out


def build_videomae(model="vit_base_patch16_224", num_classes=157, all_frames=16,
                   tubelet_size=2, img_size=224, in_chans=3, drop_path_rate=0.1,
                   head_drop_rate=0.0, init_values=0.0, finetune=None,
                   grad_checkpoint=False, verbose=True):
    """Construct a VideoMAE V2 fine-tuning ViT and optionally load weights.

    Args:
        model: one of _ARCH keys (default ViT-B/16).
        in_chans: input channels (3 for RGB, 2 for optical flow).
        finetune: path to a pretrained checkpoint (head is dropped if its
            shape disagrees with num_classes). None = random init.
        grad_checkpoint: recompute transformer blocks in backward to save VRAM.
    """
    if model not in _ARCH:
        raise ValueError(f"unknown model {model!r}; choose from {list(_ARCH)}")
    net = VisionTransformer(num_classes=num_classes, all_frames=all_frames,
                            tubelet_size=tubelet_size, img_size=img_size,
                            in_chans=in_chans, drop_path_rate=drop_path_rate,
                            head_drop_rate=head_drop_rate,
                            init_values=init_values,
                            grad_checkpoint=grad_checkpoint, **_ARCH[model])

    if finetune:
        ckpt = torch.load(finetune, map_location="cpu")
        state = _clean_state_dict(ckpt)
        # Drop the pretrained classifier head (different #classes / task).
        for k in ("head.weight", "head.bias"):
            if k in state and k in net.state_dict():
                if tuple(state[k].shape) != tuple(net.state_dict()[k].shape):
                    state.pop(k)
        # Adapt patch-embed when input channel count differs (e.g. flow 2-ch).
        pe_w = "patch_embed.proj.weight"
        if pe_w in state and pe_w in net.state_dict():
            sw, tw = state[pe_w].shape, net.state_dict()[pe_w].shape
            if sw != tw and sw[0] == tw[0] and sw[2:] == tw[2:]:
                if sw[1] == 3 and tw[1] == 2:
                    state[pe_w] = state[pe_w][:, :2].clone()
                elif sw[1] == 2 and tw[1] == 3:
                    state.pop(pe_w)
                    state.pop("patch_embed.proj.bias", None)
        msg = net.load_state_dict(state, strict=False)
        if verbose:
            missing = [m for m in msg.missing_keys if not m.startswith("pos_embed")]
            print(f"[videomae] loaded {finetune}", flush=True)
            print(f"[videomae] missing keys (expected: head + pos_embed): {missing}", flush=True)
            print(f"[videomae] unexpected keys: {msg.unexpected_keys}", flush=True)
    return net


def init_flow_backbone_from_rgb(flow_net, ckpt_path, verbose=True):
    """Copy RGB-pretrained weights into a 2-channel flow backbone."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _clean_state_dict(ckpt)
    for k in ("head.weight", "head.bias"):
        state.pop(k, None)
    target = flow_net.state_dict()
    adapted = {}
    for k, v in state.items():
        if k not in target:
            continue
        if k == "patch_embed.proj.weight" and v.shape[1] == 3 and target[k].shape[1] == 2:
            adapted[k] = v[:, :2].clone()
        elif target[k].shape == v.shape:
            adapted[k] = v
    msg = flow_net.load_state_dict(adapted, strict=False)
    if verbose:
        print(f"[videomae] flow backbone init from {ckpt_path}", flush=True)
        print(f"[videomae] flow missing: {msg.missing_keys}", flush=True)
    return flow_net


class TwoStreamVideoMAEv2(nn.Module):
    """Late-fusion two-stream VideoMAE V2 (RGB + optical flow).

    Each stream is a ViT backbone (features only). Fusion strategies match the
    CNN two-stream code: concat (MLP head), average, or learned weighted sum.
    """

    def __init__(self, rgb_net, flow_net, num_classes=157, embed_dim=768,
                 fusion="concat", proj_dim=512):
        super().__init__()
        self.fusion = fusion
        self.rgb_net = rgb_net
        self.flow_net = flow_net
        if fusion == "concat":
            self.head = nn.Sequential(
                nn.Linear(embed_dim * 2, proj_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(proj_dim, num_classes),
            )
        elif fusion == "weighted":
            self.head_rgb = nn.Linear(embed_dim, num_classes)
            self.head_flow = nn.Linear(embed_dim, num_classes)
            self.w_rgb = nn.Parameter(torch.tensor(0.0))
            self.w_flow = nn.Parameter(torch.tensor(0.0))
        elif fusion == "average":
            self.head_rgb = nn.Linear(embed_dim, num_classes)
            self.head_flow = nn.Linear(embed_dim, num_classes)
        else:
            raise ValueError(f"unknown fusion: {fusion}")

    def get_num_layers(self):
        return self.rgb_net.get_num_layers()

    def forward(self, rgb, flow):
        f_rgb = self.rgb_net.forward_features(rgb)
        f_flow = self.flow_net.forward_features(flow)
        if self.fusion == "concat":
            return self.head(torch.cat([f_rgb, f_flow], dim=1))
        logits_rgb = self.head_rgb(f_rgb)
        logits_flow = self.head_flow(f_flow)
        if self.fusion == "average":
            return (logits_rgb + logits_flow) / 2
        wr = torch.sigmoid(self.w_rgb)
        wf = torch.sigmoid(self.w_flow)
        return (wr * logits_rgb + wf * logits_flow) / (wr + wf)


def build_twostream_videomae(model="vit_base_patch16_224", num_classes=157,
                             all_frames=16, tubelet_size=2, img_size=224,
                             fusion="concat", proj_dim=512, drop_path_rate=0.1,
                             rgb_ckpt=None, flow_ckpt=None,
                             grad_checkpoint=False, verbose=True):
    """Build a late-fusion two-stream VideoMAE V2 model."""
    if model not in _ARCH:
        raise ValueError(f"unknown model {model!r}; choose from {list(_ARCH)}")
    embed_dim = _ARCH[model]["embed_dim"]
    common = dict(model=model, all_frames=all_frames, tubelet_size=tubelet_size,
                  img_size=img_size, drop_path_rate=drop_path_rate, num_classes=0,
                  grad_checkpoint=grad_checkpoint)
    rgb_net = build_videomae(in_chans=3, finetune=rgb_ckpt, verbose=verbose, **common)
    flow_net = build_videomae(in_chans=2, finetune=None, verbose=False, **common)
    ckpt = flow_ckpt or rgb_ckpt
    if ckpt:
        init_flow_backbone_from_rgb(flow_net, ckpt, verbose=verbose)
    return TwoStreamVideoMAEv2(rgb_net, flow_net, num_classes, embed_dim,
                               fusion=fusion, proj_dim=proj_dim)
