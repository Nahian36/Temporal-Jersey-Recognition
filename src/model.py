import math
from typing import Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models

from .utils import TENS_BLANK


class FrameEncoder(nn.Module):
    """Configurable lightweight encoder backed by torchvision backbones."""

    def __init__(
        self,
        embed_dim: int = 256,
        freeze_until: int = 6,
        pretrained: bool = True,
        dropout: float = 0.1,
        encoder_name: str = "mobilenet_v3_small",
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.features, in_dim = self._build_backbone(encoder_name, pretrained)
        # Freeze early layers to keep training stable and efficient
        for idx, block in enumerate(self.features):
            if idx < freeze_until:
                for p in block.parameters():
                    p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )

    def _build_backbone(self, name: str, pretrained: bool):
        name = name.lower()
        if name in {"efficientnet_lite0", "efficientnet_b0"}:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            # Classifier head is [Dropout, Linear], so grab input dim of the Linear
            in_dim = backbone.classifier[1].in_features
            return backbone.features, in_dim
        if name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            return backbone.features, 576
        if name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_large(weights=weights)
            return backbone.features, 960
        # fallback to small if unknown to stay lightweight
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        return backbone.features, 576

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class TemporalDigitNet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 192,
        num_layers: int = 1,
        dropout: float = 0.1,
        freeze_until: int = 6,
        pretrained_encoder: bool = True,
        encoder_name: str = "mobilenet_v3_small",
    ):
        super().__init__()
        self.encoder = FrameEncoder(
            embed_dim=embed_dim,
            freeze_until=freeze_until,
            pretrained=pretrained_encoder,
            dropout=dropout,
            encoder_name=encoder_name,
        )
        self.temporal = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.tens_head = nn.Linear(hidden_dim * 2, 11)  # 0-9 + blank
        self.ones_head = nn.Linear(hidden_dim * 2, 10)  # 0-9

    def forward(self, frames: torch.Tensor, mask: torch.Tensor):
        # frames: (B, T, C, H, W), mask: (B, T)
        b, t, c, h, w = frames.shape
        flat = frames.view(b * t, c, h, w)
        encoded = self.encoder(flat)
        encoded = encoded.view(b, t, -1)

        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(encoded, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.temporal(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=t)

        attn_logits = self.attn(out).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -math.inf)
        attn_weights = torch.softmax(attn_logits, dim=1)
        context = torch.sum(attn_weights.unsqueeze(-1) * out, dim=1)

        tens_logits = self.tens_head(context)
        ones_logits = self.ones_head(context)
        return tens_logits, ones_logits, attn_weights


def build_model(config: Tuple[int, int, int, float]) -> TemporalDigitNet:
    embed_dim, hidden_dim, num_layers, dropout = config
    return TemporalDigitNet(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
