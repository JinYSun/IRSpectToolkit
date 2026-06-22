"""1D-CNN光谱分类器 — PyTorch实现"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class CNN1DClassifier(nn.Module):
        """1D-CNN光谱分类器"""

        def __init__(
            self,
            n_wavelengths: int,
            n_classes: int,
            channels: list[int] = [32, 64, 128],
            kernel_sizes: list[int] = [7, 5, 3],
            dropout: float = 0.3,
        ):
            super().__init__()

            layers = []
            in_channels = 1
            for out_ch, k_size in zip(channels, kernel_sizes):
                layers.extend([
                    nn.Conv1d(in_channels, out_ch, kernel_size=k_size, padding=k_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                ])
                in_channels = out_ch

            layers.append(nn.AdaptiveAvgPool1d(8))
            self.features = nn.Sequential(*layers)

            self.classifier = nn.Sequential(
                nn.Linear(channels[-1] * 8, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            if x.ndim == 2:
                x = x.unsqueeze(1)  # (N, L) → (N, 1, L)
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

        def predict(self, x):
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                return logits.argmax(dim=1)
else:
    class CNN1DClassifier:
        """占位符 — 需要安装PyTorch"""
        def __init__(self, *args, **kwargs):
            raise ImportError("需要安装 PyTorch: pip install torch")
