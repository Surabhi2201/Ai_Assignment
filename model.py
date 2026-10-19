#!/usr/bin/env python3
"""
HAC Network Architecture for Cross-Modal Remote Sensing Retrieval.
Follows Wang et al., IEEE TGRS 2024 (Section IV-B & Fig. 2):
- Separate modality-specific AlexNet backbones for Optical (RGB) and SAR images.
- High-level 4096-D representation extracted from AlexNet fc3.
- Two parallel 1-layer fully connected projection heads per modality:
    * Euclidean head (512-D)
    * HGRMC head (512-D)
- Shared 1-layer classification head P(·) predicting 10 semantic class probabilities.
"""

import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torchvision.models as models


class AlexNetBackbone(nn.Module):
    """
    AlexNet backbone extracting 4096-D feature representation.
    Extracts features up through the 4096-D fully-connected layer (classifier[:6]).
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
        base_model = models.alexnet(weights=weights)

        # Feature extraction convolutional layers (5 conv layers + maxpools)
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # First two fully connected layers (4096-D output with ReLU and Dropout)
        self.fc_layers = base_model.classifier[:6]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)  # (N, 4096)
        return x


class HACModel(nn.Module):
    """
    Complete HAC Model with modality-specific AlexNet backbones,
    dual 512-D Euclidean and HGRMC projection heads, and shared classifier P(·).
    """
    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 512,
        pretrained: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Modality-specific AlexNet backbones
        self.optical_backbone = AlexNetBackbone(pretrained=pretrained)
        self.sar_backbone = AlexNetBackbone(pretrained=pretrained)

        # Optical projection heads: 4096 -> 512
        self.optical_euc_head = nn.Linear(4096, feature_dim)
        self.optical_hgr_head = nn.Linear(4096, feature_dim)

        # SAR projection heads: 4096 -> 512
        self.sar_euc_head = nn.Linear(4096, feature_dim)
        self.sar_hgr_head = nn.Linear(4096, feature_dim)

        # Shared 1-layer classifier P(·): 512 -> num_classes (Eq. 10)
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Initialize new projection and classifier heads reproducibly
        self._init_new_layers()

    def _init_new_layers(self):
        """
        Initialize projection heads and classifier weights.
        """
        heads = [
            self.optical_euc_head,
            self.optical_hgr_head,
            self.sar_euc_head,
            self.sar_hgr_head,
            self.classifier
        ]
        for layer in heads:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    def forward_optical(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for optical images:
        Extracts 4096-D representation, Euclidean feature f_e, HGRMC feature f_h,
        and predictions P(f_e), P(f_h).
        """
        high_level = self.optical_backbone(x)
        f_e = self.optical_euc_head(high_level)
        f_h = self.optical_hgr_head(high_level)

        pred_f_e = self.classifier(f_e)
        pred_f_h = self.classifier(f_h)

        return {
            "high_level": high_level,
            "f_e": f_e,
            "f_h": f_h,
            "pred_f_e": pred_f_e,
            "pred_f_h": pred_f_h
        }

    def forward_sar(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for SAR images:
        Extracts 4096-D representation, Euclidean feature g_e, HGRMC feature g_h,
        and predictions P(g_e), P(g_h).
        """
        high_level = self.sar_backbone(x)
        g_e = self.sar_euc_head(high_level)
        g_h = self.sar_hgr_head(high_level)

        pred_g_e = self.classifier(g_e)
        pred_g_h = self.classifier(g_h)

        return {
            "high_level": high_level,
            "g_e": g_e,
            "g_h": g_h,
            "pred_g_e": pred_g_e,
            "pred_g_h": pred_g_h
        }

    def forward(
        self,
        optical_x: torch.Tensor,
        sar_x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Joint forward pass for training pair batches.
        """
        out_opt = self.forward_optical(optical_x)
        out_sar = self.forward_sar(sar_x)

        return {
            "f_e": out_opt["f_e"],
            "f_h": out_opt["f_h"],
            "pred_f_e": out_opt["pred_f_e"],
            "pred_f_h": out_opt["pred_f_h"],
            "g_e": out_sar["g_e"],
            "g_h": out_sar["g_h"],
            "pred_g_e": out_sar["pred_g_e"],
            "pred_g_h": out_sar["pred_g_h"]
        }

    def extract_features(
        self,
        x: torch.Tensor,
        modality: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract Euclidean and HGRMC feature vectors for retrieval evaluation.
        Args:
            x: Input images tensor (B, 3, 224, 224)
            modality: 'rgb' (optical) or 'sar'
        Returns:
            (feat_euc, feat_hgr) each of shape (B, 512)
        """
        mod = modality.lower()
        if mod in ("rgb", "optical"):
            out = self.forward_optical(x)
            return out["f_e"], out["f_h"]
        elif mod == "sar":
            out = self.forward_sar(x)
            return out["g_e"], out["g_h"]
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def get_parameter_groups(
        self,
        lr_backbone: float = 0.0001,
        lr_heads: float = 0.0005,
        weight_decay: float = 0.0001
    ):
        """
        Build optimizer parameter groups according to Section VI-B:
        - Backbone layers before fc3: lr = 0.0001
        - Final projection heads and classifier: lr = 0.0005
        """
        backbone_params = []
        head_params = []

        backbone_modules = [self.optical_backbone, self.sar_backbone]
        head_modules = [
            self.optical_euc_head,
            self.optical_hgr_head,
            self.sar_euc_head,
            self.sar_hgr_head,
            self.classifier
        ]

        for m in backbone_modules:
            for p in m.parameters():
                if p.requires_grad:
                    backbone_params.append(p)

        for m in head_modules:
            for p in m.parameters():
                if p.requires_grad:
                    head_params.append(p)

        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_heads, "weight_decay": weight_decay}
        ]
