"""SignalScopeNet: dual-stream real-vs-AI detector with a generator-attribution head.

  RGB stream       : ImageNet-pretrained EfficientNet-B0 (timm), 1280-d pooled features
  Residual stream  : fixed SRM high-pass filters -> small CNN -> 128-d  (generator noise fingerprints)
  Fusion           : concat -> 256-d -> {real/fake logit, generator logits}
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import config as C

# Three classic SRM (Spatial Rich Model) residual kernels used in steganalysis / forgery detection.
_SRM = torch.tensor([
    [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
], dtype=torch.float32)
_SRM_SCALE = torch.tensor([4.0, 12.0, 2.0])


class SRMFilter(nn.Module):
    """Applies 3 fixed SRM kernels to each colour channel -> 9 residual maps. Not trainable."""

    def __init__(self):
        super().__init__()
        k = (_SRM / _SRM_SCALE[:, None, None]).unsqueeze(1)  # (3,1,5,5)
        weight = torch.zeros(9, 3, 5, 5)
        for c in range(3):
            for j in range(3):
                weight[c * 3 + j, c] = k[j, 0]
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=2)


class ResidualStream(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.srm = SRMFilter()

        def block(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.net = nn.Sequential(block(9, 32, 1), block(32, 64, 2), block(64, 128, 2), block(128, out_dim, 2))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.srm(x)
        r = torch.clamp(r, -3.0, 3.0)  # truncation, standard in SRM pipelines
        return self.pool(self.net(r)).flatten(1)


class SignalScopeNet(nn.Module):
    def __init__(self, backbone: str = C.BACKBONE, n_attr: int = len(C.ATTRIBUTION_CLASSES), pretrained: bool = True,
                 residual_dim: int = 128, drop: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.residual = ResidualStream(residual_dim)
        self.fusion = nn.Sequential(nn.Linear(feat_dim + residual_dim, 256), nn.GELU(), nn.Dropout(drop))
        self.head_bin = nn.Linear(256, 1)
        self.head_attr = nn.Linear(256, n_attr)
        self.n_attr = n_attr

    # ---- Grad-CAM target: last conv feature map of the RGB stream (7x7 for 224 input) ----
    def cam_layer(self) -> nn.Module:
        return self.backbone.conv_head if hasattr(self.backbone, "conv_head") else self.backbone.blocks[-1]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f_rgb = self.backbone(x)
        f_res = self.residual(x)
        z = self.fusion(torch.cat([f_rgb, f_res], dim=1))
        return {"logit": self.head_bin(z).squeeze(1), "attr_logits": self.head_attr(z), "embedding": z}


def build_model(pretrained: bool = True) -> SignalScopeNet:
    return SignalScopeNet(pretrained=pretrained)


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[SignalScopeNet, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {})
    model = SignalScopeNet(backbone=cfg.get("backbone", C.BACKBONE), n_attr=len(ckpt.get("attr_classes", C.ATTRIBUTION_CLASSES)), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
