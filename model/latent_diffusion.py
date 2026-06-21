import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        # timesteps shape: [B]
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / half_dim
        # freqs = e ^ (log(10000) * exponent)
        freqs = torch.exp(-np.log(10000.0) * exponent)
        args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class DenoisingNetwork(nn.Module):
    """
    MLP-ResNet architecture representing the Denoising Network (equivalent to DiT for 1D vectors).
    Takes noisy latent z_t, timestep t, and condition y.
    """
    def __init__(self, latent_dim=16, embed_dim=64, num_classes=2):
        super().__init__()
        self.time_embed = nn.Sequential(
            TimestepEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        
        # Input layer mapping latent_dim to embed_dim
        self.in_proj = nn.Linear(latent_dim, embed_dim)
        
        # ResNet blocks
        self.block1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.block2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Output project back to latent_dim
        self.out_proj = nn.Linear(embed_dim, latent_dim)

    def forward(self, z_t, t, y):
        # z_t: [B, latent_dim]
        # t: [B]
        # y: [B]
        
        # Project inputs
        z_emb = self.in_proj(z_t) # [B, embed_dim]
        t_emb = self.time_embed(t) # [B, embed_dim]
        y_emb = self.class_embed(y) # [B, embed_dim]
        
        # Condition blending: add time and class embeddings
        h = z_emb + t_emb + y_emb
        
        # ResNet blocks with skip connections
        h1 = self.block1(h)
        h = h + h1
        
        h2 = self.block2(h)
        h = h + h2
        
        # Output prediction
        out = self.out_proj(h)
        return out


class LatentDiffusion(nn.Module):
    """
    Diffusion Model operating in VAE Latent Space.
    Implements standard DDPM (Denoising Diffusion Probabilistic Models) equations.
    """
    def __init__(self, latent_dim=16, steps=100, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.latent_dim = latent_dim
        self.steps = steps
        
        # Denoising model
        self.denoiser = DenoisingNetwork(latent_dim=latent_dim, embed_dim=64, num_classes=2)
        
        # Diffusion schedules
        betas = torch.linspace(beta_start, beta_end, steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Buffers for forward diffusion
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(z_t | z_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Calculations for reverse process posterior q(z_{t-1} | z_t, z_0)
        # sigma_t^2 = beta_t * (1 - alpha_cumprod_{t-1}) / (1 - alpha_cumprod_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def q_sample(self, z_0, t, noise=None):
        """Forward diffusion step: add noise to z_0 at step t."""
        if noise is None:
            noise = torch.randn_like(z_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * z_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def training_loss(self, z_0, y):
        """Compute MSE loss for training."""
        B = z_0.size(0)
        # Sample a random timestep for each trial in batch
        t = torch.randint(0, self.steps, (B,), device=z_0.device).long()
        noise = torch.randn_like(z_0)
        
        # Get noisy latent z_t
        z_t = self.q_sample(z_0, t, noise=noise)
        
        # Predict noise
        noise_pred = self.denoiser(z_t, t.float(), y)
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, z_t, t, y):
        """Single reverse denoising step."""
        # Predict noise
        noise_pred = self.denoiser(z_t, torch.tensor([t], device=z_t.device).float().repeat(z_t.size(0)), y)
        
        # Compute mean
        # z_{t-1} mean = 1 / sqrt(alpha_t) * (z_t - beta_t / sqrt(1 - alpha_cumprod_t) * noise_pred)
        alpha_t = self.alphas[t]
        beta_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        mean = (1.0 / torch.sqrt(alpha_t)) * (z_t - (beta_t / sqrt_one_minus_alphas_cumprod_t) * noise_pred)
        
        if t == 0:
            return mean
        else:
            # Add Langevin noise
            variance = self.posterior_variance[t]
            noise = torch.randn_like(z_t)
            return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, num_samples, condition_label, vae_model, device='cpu'):
        """Sample from the diffusion model and decode using the VAE."""
        self.eval()
        vae_model.eval()
        
        # Start from pure Gaussian noise
        z = torch.randn(num_samples, self.latent_dim, device=device)
        y = torch.tensor([condition_label], device=device).repeat(num_samples)
        
        # Reverse denoising
        for t in reversed(range(self.steps)):
            z = self.p_sample(z, t, y)
            
        # Decode sampled latents to signal space
        sampled_signals = vae_model.decode(z)
        
        return sampled_signals, z
