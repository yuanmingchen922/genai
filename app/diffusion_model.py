"""
Diffusion Model Implementation for CIFAR-10
Implements DDPM (Denoising Diffusion Probabilistic Model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal Time Embedding for diffusion timesteps.
    
    Formula for i-th dimension:
    - If i is even: sin(t / (10000^(2i/d)))
    - If i is odd:  cos(t / (10000^(2i/d)))
    
    Where:
    - t: timestep
    - d: embedding dimension
    - i: dimension index (0 to d-1)
    """
    
    def __init__(self, embedding_dim=128, max_period=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
    
    def forward(self, timesteps):
        """
        Args:
            timesteps: (batch_size,) tensor of timestep indices
        
        Returns:
            (batch_size, embedding_dim) tensor of embeddings
        """
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        
        # Calculate frequency scaling: 10000^(-2i/d) for i in [0, d/2)
        # This is equivalent to exp(-2i * log(10000) / d)
        frequencies = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=device) / half_dim
        )
        
        # Compute arguments: t * frequency
        args = timesteps[:, None].float() * frequencies[None, :]
        
        # Concatenate sin and cos components
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding


class ResidualBlock(nn.Module):
    """Residual block with time embedding injection."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb):
        """
        Args:
            x: (B, C, H, W) input tensor
            time_emb: (B, time_emb_dim) time embedding
        """
        residual = x
        
        # First conv block
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        
        # Inject time embedding
        time_emb = self.time_mlp(F.relu(time_emb))
        x = x + time_emb[:, :, None, None]
        
        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)
        x = F.relu(x)
        
        # Residual connection
        return x + self.residual_conv(residual)


class AttentionBlock(nn.Module):
    """Self-attention block for spatial features."""
    
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # Reshape for attention
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = k.reshape(B, C, H * W)                    # (B, C, HW)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        
        # Attention
        attn = torch.bmm(q, k) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(attn, v)  # (B, HW, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        
        out = self.proj(out)
        return out + residual


class UNet(nn.Module):
    """
    UNet architecture for Diffusion Model.
    Predicts noise to remove from noisy image at timestep t.
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_emb_dim=128,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        use_attention=(False, False, True, True)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        channels = [base_channels]
        in_ch = base_channels
        
        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            
            # Residual blocks at this level
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(in_ch, out_ch, time_emb_dim * 4)
                )
                if use_attention[level]:
                    self.down_blocks.append(AttentionBlock(out_ch))
                channels.append(out_ch)
                in_ch = out_ch
            
            # Downsample (except last level)
            if level < len(channel_multipliers) - 1:
                self.down_samples.append(
                    nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
                )
                channels.append(out_ch)
            else:
                self.down_samples.append(nn.Identity())
        
        # Bottleneck
        self.mid_block1 = ResidualBlock(out_ch, out_ch, time_emb_dim * 4)
        self.mid_attn = AttentionBlock(out_ch)
        self.mid_block2 = ResidualBlock(out_ch, out_ch, time_emb_dim * 4)
        
        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                self.up_blocks.append(
                    ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim * 4)
                )
                if use_attention[-(level + 1)]:
                    self.up_blocks.append(AttentionBlock(out_ch))
                in_ch = out_ch
            
            # Upsample (except last level)
            if level < len(channel_multipliers) - 1:
                self.up_samples.append(
                    nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1)
                )
            else:
                self.up_samples.append(nn.Identity())
        
        # Final output
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def forward(self, x, t):
        """
        Args:
            x: (B, C, H, W) noisy image at timestep t
            t: (B,) timestep indices
        
        Returns:
            (B, C, H, W) predicted noise
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Initial conv
        x = self.init_conv(x)
        
        # Store skip connections
        skip_connections = [x]
        
        # Encoder
        block_idx = 0
        for downsample in self.down_samples:
            # Apply residual blocks at this level
            while block_idx < len(self.down_blocks) and not isinstance(
                self.down_blocks[block_idx], type(downsample)
            ):
                x = self.down_blocks[block_idx](x, t_emb) if isinstance(
                    self.down_blocks[block_idx], ResidualBlock
                ) else self.down_blocks[block_idx](x)
                skip_connections.append(x)
                block_idx += 1
            
            # Downsample
            x = downsample(x)
            if not isinstance(downsample, nn.Identity):
                skip_connections.append(x)
        
        # Bottleneck
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        
        # Decoder
        block_idx = 0
        for upsample in self.up_samples:
            # Apply residual blocks with skip connections
            for _ in range(3):  # num_res_blocks + 1
                skip = skip_connections.pop()
                x = torch.cat([x, skip], dim=1)
                x = self.up_blocks[block_idx](x, t_emb) if isinstance(
                    self.up_blocks[block_idx], ResidualBlock
                ) else self.up_blocks[block_idx](x)
                block_idx += 1
                
                # Check for attention block
                if block_idx < len(self.up_blocks) and isinstance(
                    self.up_blocks[block_idx], AttentionBlock
                ):
                    x = self.up_blocks[block_idx](x)
                    block_idx += 1
            
            # Upsample
            x = upsample(x)
        
        # Final output
        x = self.final_norm(x)
        x = F.relu(x)
        x = self.final_conv(x)
        
        return x


class DiffusionModel:
    """
    Complete Diffusion Model with forward/reverse diffusion process.
    Implements DDPM (Denoising Diffusion Probabilistic Models).
    """
    
    def __init__(
        self,
        model: nn.Module,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        device='cpu'
    ):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values for diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Values for reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: add noise to image.
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        
        Args:
            x_0: (B, C, H, W) original image
            t: (B,) timestep indices
            noise: optional noise tensor
        
        Returns:
            (B, C, H, W) noisy image at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    def p_losses(self, x_0, t, noise=None):
        """
        Calculate training loss (MSE between predicted and actual noise).
        
        Args:
            x_0: (B, C, H, W) original image
            t: (B,) timestep indices
            noise: optional noise tensor
        
        Returns:
            loss scalar
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get noisy image
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Reverse diffusion: denoise one step.
        
        Args:
            x_t: (B, C, H, W) noisy image at timestep t
            t: (B,) timestep indices
        
        Returns:
            (B, C, H, W) image at timestep t-1
        """
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Calculate x_{t-1}
        sqrt_recip_alpha = self.sqrt_recip_alphas[t][:, None, None, None]
        beta = self.betas[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        # Mean of reverse distribution
        mean = sqrt_recip_alpha * (x_t - beta * predicted_noise / sqrt_one_minus_alpha_bar)
        
        # Add noise (except for t=0)
        if t[0] > 0:
            variance = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean
    
    @torch.no_grad()
    def sample(self, batch_size, channels, height, width):
        """
        Generate samples by reverse diffusion from pure noise.
        
        Args:
            batch_size: number of samples
            channels, height, width: image dimensions
        
        Returns:
            (B, C, H, W) generated images
        """
        self.model.eval()
        
        # Start from pure noise
        x = torch.randn(batch_size, channels, height, width).to(self.device)
        
        # Reverse diffusion
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            x = self.p_sample(x, t_batch)
        
        return x


# Simplified UNet for easier use
class SimpleUNet(nn.Module):
    """Simplified UNet for CIFAR-10."""
    
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
        )
        
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down1 = ResidualBlock(64, 128, time_emb_dim * 4)
        self.down2 = ResidualBlock(128, 256, time_emb_dim * 4)
        self.downsample1 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        
        self.mid1 = ResidualBlock(256, 256, time_emb_dim * 4)
        self.mid_attn = AttentionBlock(256)
        self.mid2 = ResidualBlock(256, 256, time_emb_dim * 4)
        
        self.up1 = ResidualBlock(512, 128, time_emb_dim * 4)
        self.up2 = ResidualBlock(256, 64, time_emb_dim * 4)
        self.upsample1 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        
        self.final_norm = nn.GroupNorm(8, 64)
        self.final_conv = nn.Conv2d(64, out_channels, 3, padding=1)
    
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        x = self.init_conv(x)
        skip0 = x
        
        x = self.down1(x, t_emb)
        skip1 = x
        x = self.downsample1(x)
        
        x = self.down2(x, t_emb)
        skip2 = x
        x = self.downsample2(x)
        
        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)
        
        x = self.upsample1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up1(x, t_emb)
        
        x = self.upsample2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up2(x, t_emb)
        
        x = self.final_norm(x)
        x = F.relu(x)
        x = self.final_conv(x)
        
        return x


def get_diffusion_model(device='cpu'):
    """Create and return a Diffusion Model instance."""
    unet = SimpleUNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=128
    ).to(device)
    
    diffusion = DiffusionModel(
        model=unet,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        device=device
    )
    
    return diffusion

