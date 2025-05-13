import torch
import torch.nn as nn

# transformer model that maps mel spectrograms to pca-reduced expression offsets
class AudioToExpressionTransformer(nn.Module):
    def __init__(self, 
                 input_dim=464,
                 output_dim=284,         # updated to match PCA-reduced target
                 hidden_dim=512,         # wider feedforward dimension for better capacity
                 num_heads=8,
                 num_layers=2,
                 dropout=0.5,
                 max_seq_len=30):        # shorter sequences due to windowing

        super().__init__()

        # linear projection for input features
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # learnable positional encoding
        self.positional_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim)
        )

        # normalization before transformer
        self.norm = nn.LayerNorm(hidden_dim)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # final projection to expression space
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, mel_input):
        # handle both [B, T, 16, 29] and [B, T, 464] input formats
        if mel_input.ndim == 4:
            B, T, C, F = mel_input.shape
            mel_input = mel_input.view(B, T, C * F)
        elif mel_input.ndim != 3:
            raise ValueError(f"expected 3D or 4D tensor, got {mel_input.shape}")

        B, T, _ = mel_input.shape

        # projection + position encoding
        x = self.input_proj(mel_input)
        x = x + self.positional_embedding[:, :T, :]
        x = self.norm(x)

        # transformer forward pass
        x = self.encoder(x)

        # project to expression offset space
        return self.output_proj(x)


