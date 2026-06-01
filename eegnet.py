"""EEGNet — Lawhern et al. (2018) compact CNN for EEG classification.

Single-file, self-contained model definition. No external dependencies beyond
torch. See the README for full architecture and training details.

Architecture follows Table 2 of the original paper:
  - Temporal kernel = srate // 2  (captures ≥2 Hz)
  - Separable kernel = srate // 8 (captures 500ms at effective rate after pool)
  - Dropout = 0.25 for cross-subject classification
  - Max-norm constraints: depthwise conv = 1.0, classifier = 0.25
"""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet for binary EEG classification.

    Input:  (B, C, T)  — batch of EEG epochs
    Output: (B, 1)     — logit (use BCEWithLogitsLoss)
    """

    def __init__(self, n_channels: int, n_times: int, srate: int = 256,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 dropout: float = 0.25):
        super().__init__()
        self.n_channels = n_channels

        # Kernel sizes scaled to sampling rate (paper Table 2, Section 2.2.1)
        kernel_length = srate // 2     # half the sampling rate → captures ≥2 Hz
        sep_kernel = srate // 8        # 500ms at effective rate after pool(4)

        # --- Block 1: temporal conv + depthwise spatial conv ---
        self.conv_temporal = nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv_depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # --- Block 2: separable conv ---
        self.conv_sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, sep_kernel), groups=F1 * D, padding="same", bias=False)
        self.conv_sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # --- Classifier ---
        # Compute flat size via a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            dummy = self.pool1(self.bn2(self.conv_depthwise(self.bn1(self.conv_temporal(dummy)))))
            dummy = self.pool2(self.bn3(self.conv_sep_point(self.conv_sep_depth(dummy))))
            flat_size = dummy.numel()

        self.classifier = nn.Linear(flat_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, 1, C, T)
        x = x.unsqueeze(1)

        # Block 1
        x = self.conv_temporal(x)
        x = self.bn1(x)
        x = self.conv_depthwise(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv_sep_depth(x)
        x = self.conv_sep_point(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Classify
        x = x.flatten(1)
        return self.classifier(x)

    def apply_weight_constraint(self, depthwise_max_norm: float = 1.0,
                                classifier_max_norm: float = 0.25):
        """Clamp weight norms per paper Table 2 (called after each optimizer step).

        - Depthwise conv: max_norm = 1.0
        - Classifier dense: max_norm = 0.25
        """
        # Depthwise conv constraint
        w = self.conv_depthwise.weight.data
        norm = w.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        desired = torch.clamp(norm, max=depthwise_max_norm)
        w.mul_(desired / (norm + 1e-8))

        # Classifier constraint
        w_cls = self.classifier.weight.data
        norm_cls = w_cls.norm(dim=1, keepdim=True)
        desired_cls = torch.clamp(norm_cls, max=classifier_max_norm)
        w_cls.mul_(desired_cls / (norm_cls + 1e-8))
