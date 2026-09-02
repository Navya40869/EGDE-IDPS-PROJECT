"""
cnn_model.py — Phase 4: CNN Architecture

Input -> Conv1D -> BatchNorm -> ReLU -> MaxPool -> Dropout
      -> Conv1D -> GlobalAveragePooling -> Fully Connected -> Softmax

Input shape: (batch_size, 1, num_features) — features treated as a 1D sequence.
"""

import torch
import torch.nn as nn


class EdgeIDPSCNN(nn.Module):
    def __init__(self, num_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        # No explicit Softmax here — CrossEntropyLoss applies log-softmax internally.
        # Apply torch.softmax(logits, dim=1) at inference time for probabilities.

        self.num_features = num_features
        self.num_classes = num_classes

    def forward(self, x):
        # x expected shape: (batch, num_features) -> reshape to (batch, 1, num_features)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.global_avg_pool(x)  # (batch, 64, 1)
        x = x.squeeze(-1)            # (batch, 64)
        logits = self.fc(x)          # (batch, num_classes)
        return logits


if __name__ == "__main__":
    # Quick sanity check
    model = EdgeIDPSCNN(num_features=20, num_classes=8)
    dummy_input = torch.randn(4, 20)  # batch of 4
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # expect (4, 8)
    print(model)