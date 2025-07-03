import torch
import torch.nn as nn

class DriftTransformer(nn.Module):
    def __init__(self, input_dim=4, model_dim=128, nhead=8, num_layers=4, output_dim=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)           # shape: [B, T, D]
        x = self.transformer(x)          # shape: [B, T, D]
        x = x.mean(dim=1)                # 平均池化
        return self.output(x)            # shape: [B, 2]