"""Two-stream networks for multi-label Charades (157 logits, no softmax).

- SpatialStream  : ResNet-50 on RGB frames, temporal average pooling.
- TemporalStream : ResNet-50 with conv1 inflated to 2*num_flow input channels.
- TwoStreamFusion: late fusion of the two streams (average / weighted / concat).

All classifiers output raw logits; apply sigmoid for per-class probabilities.
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torchvision.models as models


def _resnet50_backbone():
    bb = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    feat_dim = bb.fc.in_features                      # 2048
    backbone = nn.Sequential(*list(bb.children())[:-1])  # -> [B,2048,1,1]
    return backbone, feat_dim


class SpatialStream(nn.Module):
    """RGB stream. Input [B, T, 3, H, W]. Processes frames in chunks to limit VRAM."""

    def __init__(self, num_classes=157, proj_dim=512, temporal_chunk=16,
                 grad_checkpoint=False):
        super().__init__()
        self.temporal_chunk = temporal_chunk
        self.grad_checkpoint = grad_checkpoint
        self.backbone, c = _resnet50_backbone()
        self.proj = nn.Sequential(nn.Dropout(0.5), nn.Linear(c, proj_dim), nn.ReLU(inplace=True))
        self.clf = nn.Linear(proj_dim, num_classes)

    def _run_backbone(self, inp):
        # Checkpoint each chunk during training: recompute activations in the
        # backward pass instead of storing them (needed for very long clips).
        if self.grad_checkpoint and self.training:
            return cp.checkpoint(self.backbone, inp, use_reentrant=False)
        return self.backbone(inp)

    def forward(self, x, feats_only=False):
        b, t, c, h, w = x.shape
        parts = []
        for s in range(0, t, self.temporal_chunk):
            chunk = x[:, s:s + self.temporal_chunk]
            tc = chunk.shape[1]
            f = self._run_backbone(chunk.reshape(b * tc, c, h, w)).flatten(1)
            parts.append(f.view(b, tc, -1))
        f = torch.cat(parts, dim=1).mean(1)
        f = self.proj(f)
        return f if feats_only else self.clf(f)


class TemporalStream(nn.Module):
    """Optical-flow stream. Input [B, 2*T, H, W]. Chunked like SpatialStream."""

    def __init__(self, num_classes=157, in_ch=20, proj_dim=512, temporal_chunk=16,
                 grad_checkpoint=False):
        super().__init__()
        self.temporal_chunk = temporal_chunk
        self.grad_checkpoint = grad_checkpoint
        self.backbone, c = _resnet50_backbone()
        orig = self.backbone[0]
        new_conv = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            avg_w = orig.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(avg_w.repeat(1, in_ch, 1, 1) / in_ch)
        self.backbone[0] = new_conv
        self.proj = nn.Sequential(nn.Dropout(0.5), nn.Linear(c, proj_dim), nn.ReLU(inplace=True))
        self.clf = nn.Linear(proj_dim, num_classes)

    def forward(self, x, feats_only=False):
        b, c, h, w = x.shape
        t = c // 2
        x = x.view(b, t, 2, h, w)
        parts = []
        for s in range(0, t, self.temporal_chunk):
            chunk = x[:, s:s + self.temporal_chunk]
            tc = chunk.shape[1]
            inp = chunk.reshape(b, tc * 2, h, w)
            if tc < self.temporal_chunk:
                pad = torch.zeros(b, (self.temporal_chunk - tc) * 2, h, w,
                                  device=inp.device, dtype=inp.dtype)
                inp = torch.cat([inp, pad], dim=1)
            feat = (cp.checkpoint(self.backbone, inp, use_reentrant=False)
                    if self.grad_checkpoint and self.training else self.backbone(inp))
            parts.append(feat.flatten(1))
        f = torch.stack(parts, dim=1).mean(1)
        f = self.proj(f)
        return f if feats_only else self.clf(f)


class TwoStreamFusion(nn.Module):
    """Late fusion. fusion in {'average','weighted','concat'}."""

    def __init__(self, num_classes=157, proj_dim=512, flow_ch=20, fusion="concat",
                 temporal_chunk=16, grad_checkpoint=False):
        super().__init__()
        self.fusion = fusion
        self.spatial = SpatialStream(num_classes, proj_dim, temporal_chunk, grad_checkpoint)
        flow_ch = 2 * temporal_chunk
        self.temporal = TemporalStream(num_classes, flow_ch, proj_dim, temporal_chunk,
                                       grad_checkpoint)
        if fusion == "weighted":
            self.w_rgb = nn.Parameter(torch.tensor(0.0))
            self.w_flow = nn.Parameter(torch.tensor(0.0))
        elif fusion == "concat":
            self.head = nn.Sequential(
                nn.Linear(proj_dim * 2, proj_dim), nn.ReLU(inplace=True),
                nn.Dropout(0.3), nn.Linear(proj_dim, num_classes),
            )
        elif fusion != "average":
            raise ValueError(f"unknown fusion: {fusion}")

    def forward(self, rgb, flow):
        if self.fusion == "concat":
            f = torch.cat([self.spatial(rgb, feats_only=True),
                           self.temporal(flow, feats_only=True)], dim=1)
            return self.head(f)
        sr, tr = self.spatial(rgb), self.temporal(flow)
        if self.fusion == "average":
            return (sr + tr) / 2
        wr, wf = torch.sigmoid(self.w_rgb), torch.sigmoid(self.w_flow)
        return (wr * sr + wf * tr) / (wr + wf)


def build_model(stream, num_classes=157, proj_dim=512, num_flow=10, fusion="concat",
                temporal_chunk=16, grad_checkpoint=False):
    """stream in {'rgb','flow','fusion'}. Flow/fusion conv1 sized for temporal_chunk pairs.

    grad_checkpoint=True trades compute for memory (recomputes backbone activations
    in the backward pass) — needed for very long clips (high-fps subsets).
    """
    flow_ch = 2 * temporal_chunk
    if stream == "rgb":
        return SpatialStream(num_classes, proj_dim, temporal_chunk, grad_checkpoint)
    if stream == "flow":
        return TemporalStream(num_classes, flow_ch, proj_dim, temporal_chunk, grad_checkpoint)
    if stream == "fusion":
        return TwoStreamFusion(num_classes, proj_dim, flow_ch, fusion, temporal_chunk,
                               grad_checkpoint)
    raise ValueError(f"unknown stream: {stream}")
