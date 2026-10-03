"""
src/task2/classifier.py
-----------------------
Corruption classifier architectures for Task 2 (Hard-Routing Restoration System).

Classifies 128x128 RGB images into 4 corruption categories:
    0: Clean (Identity bypass)
    1: Salt-and-Pepper noise
    2: Gaussian blur
    3: Rectangular occlusion

Architectural options supported:
    1. Custom Conv Stack (CustomConvClassifier): 4 conv blocks + GAP + Linear head.
    2. MobileNet-style (MobileNetClassifier): Depthwise separable inverted residuals + GAP + Linear head.
    3. ResNet-18 (ResNet18Classifier): ResNet-18 adapted for 4-class output.
    4. CorruptionClassifier: Unified facade wrapping any of the three backbones.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CustomConvClassifier(nn.Module):
    """4-stage convolutional neural network with BatchNorm, LeakyReLU, and GAP."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256),
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = tuple(channels)
        self.num_classes = num_classes
        self.dropout_rate = dropout

        stages: list[nn.Module] = []
        current_c = in_channels
        for out_c in self.channels:
            stages.extend(
                [
                    nn.Conv2d(current_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            current_c = out_c

        self.features = nn.Sequential(*stages)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(current_c, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.gap(feat)
        return torch.flatten(feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        feat = self.dropout(feat)
        return self.fc(feat)


class InvertedResidualBlock(nn.Module):
    """Depthwise separable inverted residual block inspired by MobileNetV2."""

    def __init__(self, in_c: int, out_c: int, stride: int, expand_ratio: int = 2) -> None:
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_c == out_c
        hidden_dim = int(round(in_c * expand_ratio))

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_c, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise linear
            nn.Conv2d(hidden_dim, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetClassifier(nn.Module):
    """Compact depthwise-separable convolutional classifier."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 96, 160),
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = tuple(channels)
        self.num_classes = num_classes
        self.dropout_rate = dropout

        first_c = channels[0]
        # Stem conv: 128x128 -> 64x64
        stages: list[nn.Module] = [
            nn.Conv2d(in_channels, first_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(first_c),
            nn.ReLU6(inplace=True),
        ]

        curr_c = first_c
        for next_c in channels[1:]:
            # Downsample block then same-resolution block
            stages.append(InvertedResidualBlock(curr_c, next_c, stride=2, expand_ratio=2))
            stages.append(InvertedResidualBlock(next_c, next_c, stride=1, expand_ratio=2))
            curr_c = next_c

        self.features = nn.Sequential(*stages)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(curr_c, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.gap(feat)
        return torch.flatten(feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        feat = self.dropout(feat)
        return self.fc(feat)


class ResNet18Classifier(nn.Module):
    """ResNet-18 backbone adapted for 4-class corruption classification."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        dropout: float = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)

        if in_channels != 3:
            base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Backbone without fc
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.gap = base.avgpool
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        feat = self.dropout(feat)
        return self.fc(feat)


class CorruptionClassifier(nn.Module):
    """Unified Corruption Classifier facade for Task 2."""

    def __init__(
        self,
        backbone: str = "custom_conv",
        in_channels: int = 3,
        num_classes: int = 4,
        channels: Optional[Sequence[int]] = None,
        dropout: float = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone.lower()
        self.num_classes = num_classes

        if self.backbone_name == "custom_conv":
            c = channels or (32, 64, 128, 256)
            self.model: nn.Module = CustomConvClassifier(
                in_channels=in_channels, channels=c, num_classes=num_classes, dropout=dropout
            )
        elif self.backbone_name == "mobilenet":
            c = channels or (32, 64, 96, 160)
            self.model = MobileNetClassifier(
                in_channels=in_channels, channels=c, num_classes=num_classes, dropout=dropout
            )
        elif self.backbone_name in ("resnet18", "resnet"):
            self.model = ResNet18Classifier(
                in_channels=in_channels, num_classes=num_classes, dropout=dropout, pretrained=pretrained
            )
        else:
            raise ValueError(
                f"Unknown backbone: '{backbone}'. Choose from 'custom_conv', 'mobilenet', 'resnet18'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized logits (B, num_classes)."""
        return self.model(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized class probabilities via Softmax (B, num_classes)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class index via Argmax (B,)."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled bottleneck feature vector (B, D)."""
        return self.model.extract_features(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_classifier(
    backbone: str = "custom_conv",
    in_channels: int = 3,
    num_classes: int = 4,
    channels: Optional[Sequence[int]] = None,
    dropout: float = 0.2,
    pretrained: bool = False,
) -> CorruptionClassifier:
    """Helper factory for constructing a CorruptionClassifier instance."""
    return CorruptionClassifier(
        backbone=backbone,
        in_channels=in_channels,
        num_classes=num_classes,
        channels=channels,
        dropout=dropout,
        pretrained=pretrained,
    )
