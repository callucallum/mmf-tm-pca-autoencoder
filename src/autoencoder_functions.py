import torch
import torch.nn as nn

EPS = 1e-12


def phase_aligned_loss_wholeTM_perSample(pred_TM, true_TM):
    B = pred_TM.shape[0]
    pred_flat = pred_TM.reshape(B, -1)
    true_flat = true_TM.reshape(B, -1)

    inner = torch.sum(pred_flat * torch.conj(true_flat), dim=1)
    phase = torch.angle(inner).unsqueeze(1)
    pred_aligned = pred_flat * torch.exp(-1j * phase)

    diff = true_flat - pred_aligned
    diff_norm = torch.linalg.norm(diff, dim=1)
    true_norm = torch.linalg.norm(true_flat, dim=1).clamp_min(EPS)
    rel_err = diff_norm / true_norm

    # Add amplitude consistency term to prevent trivial zero solution
    pred_norm = torch.linalg.norm(pred_flat, dim=1).clamp_min(EPS)
    amp_ratio = pred_norm / true_norm
    amp_penalty = (amp_ratio - 1).abs().mean()

    return rel_err.mean() + 0 * amp_penalty, rel_err.mean(), amp_penalty


class ChannelAttnLinformer(nn.Module):
    def __init__(self, latent_dim=28*4, H=4, W=4, num_heads=2, k=14):
        super().__init__()
        self.latent_dim = latent_dim
        self.H = H
        self.W = W
        self.embed_dim = H * W  # 16
        self.k = k  # projected sequence length

        # Learnable projections for K and V (sequence length: C -> k)
        self.E = nn.Linear(latent_dim, k)  # projects K
        self.F = nn.Linear(latent_dim, k)  # projects V

        self.attn = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=False)

    def forward(self, x):
        # x: [B*N, C=latent_dim, H, W]
        BN, C, H, W = x.shape

        # [B*N, C, H*W]
        tokens = x.view(BN, C, H * W)
        # [C, BN, H*W]
        tokens_T = tokens.permute(1, 0, 2)

        # Q stays full length [C, BN, H*W]
        Q = tokens_T

        # Project K and V from C -> k along sequence dimension
        # E and F expect [..., C] so we permute to [BN, H*W, C] then back
        tokens_for_proj = tokens_T.permute(1, 2, 0)  # [BN, H*W, C]
        K = self.E(tokens_for_proj).permute(2, 0, 1)  # [k, BN, H*W]
        V = self.F(tokens_for_proj).permute(2, 0, 1)  # [k, BN, H*W]

        attn_out, _ = self.attn(Q, K, V)  # [C, BN, H*W]

        tokens_out = attn_out.permute(1, 0, 2)   # [BN, C, H*W]
        x_out = tokens_out.view(BN, C, H, W)
        return x_out

class TM_Autoencoder(nn.Module):
    def __init__(self, input_channels=2, enc_channels=[16, 32, 48], latent_dim=8,
                 attn_heads=1, linformer_k=16):
        super().__init__()

        # Encoder: 208 -> 104 -> 52 -> 26 -> 13
        self.enc = nn.Sequential(
            nn.Conv2d(input_channels,      enc_channels[0], 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(enc_channels[0], enc_channels[1], 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(enc_channels[1], enc_channels[2], 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(enc_channels[2], enc_channels[2], 3, stride=2, padding=1), nn.GELU(),
        )

        # Channel attention at 13x13 bottleneck
        self.channel_attn = ChannelAttnLinformer(
            latent_dim=enc_channels[2], H=13, W=13,
            num_heads=attn_heads, k=linformer_k
        )

        # 1x1 conv to compress channels -> latent_dim
        self.to_latent = nn.Conv2d(enc_channels[2], latent_dim, 1)

        # 1x1 conv to expand back
        self.from_latent = nn.Conv2d(latent_dim, enc_channels[2], 1)

        # Decoder: 13 -> 26 -> 52 -> 104 -> 208
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(enc_channels[2], enc_channels[2], 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(enc_channels[2], enc_channels[1], 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(enc_channels[1], enc_channels[0], 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(enc_channels[0], input_channels,  4, stride=2, padding=1),
        )

        self._printed_shapes = False

    def forward(self, x, show_shapes=True):
        B, C, N, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B*N, C, H, W)

        # Encode
        f = self.enc(x_flat)                    # [B*N, 48, 13, 13]

        # Channel attention + residual
        f_attn = self.channel_attn(f)
        f_res = f + f_attn                       # [B*N, 48, 13, 13]

        # Compress to latent
        latent_e = self.to_latent(f_res)         # [B*N, latent_dim, 13, 13]
        latent = latent_e.view(B, N, latent_e.shape[1],
                               latent_e.shape[2], latent_e.shape[3])

        # Expand from latent
        latent_d = latent.view(B*N, latent_e.shape[1], 13, 13)
        z = self.from_latent(latent_d)           # [B*N, 48, 13, 13]

        # Decode
        x_rec = self.dec(z)                      # [B*N, C, 208, 208]
        x_final = x_rec.view(B, N, C, H, W).permute(0, 2, 1, 3, 4)

        if show_shapes and not self._printed_shapes:
            self._printed_shapes = True
            print(f"Input:   {x.shape}")
            print(f"Encoded: {f.shape}")
            print(f"Latent:  {latent.shape}")
            print(f"Output:  {x_final.shape}")

        return x_final, latent
    