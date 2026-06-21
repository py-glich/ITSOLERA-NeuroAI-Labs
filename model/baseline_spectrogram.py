import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpectrogramUNet(nn.Module):
    """
    A simple 2D U-Net style denoiser for EEG Spectrograms.
    Input shape: [B, C*2, Freqs, TimeFrames] where 2 channels represent Real and Imaginary parts.
    """
    def __init__(self, in_channels=16, out_channels=16, embed_dim=32, num_classes=2):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        
        # Down-blocks
        self.down1 = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, stride=2, padding=1) # [B, 64, F/2, T/2]
        
        # Bottleneck
        self.middle = nn.Conv2d(embed_dim * 2, embed_dim * 2, kernel_size=3, padding=1)
        
        # Up-blocks
        self.up1 = nn.ConvTranspose2d(embed_dim * 2, embed_dim, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.Conv2d(embed_dim * 2, out_channels, kernel_size=3, padding=1) # concat skip

    def forward(self, x, t, y):
        # x: [B, in_channels, Freqs, TimeFrames]
        # t: [B]
        # y: [B]
        
        B, C, F_len, T_len = x.shape
        
        # Embeddings
        t_emb = self.time_embed(t.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1) # [B, embed_dim, 1, 1]
        y_emb = self.class_embed(y).unsqueeze(-1).unsqueeze(-1) # [B, embed_dim, 1, 1]
        cond = t_emb + y_emb # [B, embed_dim, 1, 1]
        
        # Encoder
        h1 = F.gelu(self.down1(x) + cond) # [B, embed_dim, Freqs, TimeFrames]
        h2 = F.gelu(self.down2(h1)) # [B, embed_dim*2, Freqs/2, TimeFrames/2]
        
        # Bottleneck
        h_mid = F.gelu(self.middle(h2))
        
        # Decoder
        h_up = F.gelu(self.up1(h_mid)) # [B, embed_dim, Freqs, TimeFrames]
        
        # Size matching (in case of odd dimensions)
        if h_up.shape[-2:] != h1.shape[-2:]:
            h_up = F.interpolate(h_up, size=h1.shape[-2:], mode='bilinear', align_corners=False)
            
        h_concat = torch.cat([h_up, h1], dim=1) # [B, embed_dim * 2, Freqs, TimeFrames]
        out = self.up2(h_concat)
        
        # Crop/interpolate output to match original input size exactly
        if out.shape[-2:] != (F_len, T_len):
            out = F.interpolate(out, size=(F_len, T_len), mode='bilinear', align_corners=False)
            
        return out


class BaselineSpectrogramDiffusion(nn.Module):
    """
    Baseline 2D Spectrogram-based Diffusion Model.
    EEG 1D -> STFT -> 2D Spectrogram -> 2D DDPM Diffusion -> iSTFT -> EEG 1D
    """
    def __init__(self, num_channels=8, seq_len=500, steps=100, sfreq=250, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.steps = steps
        self.sfreq = sfreq
        
        # STFT configuration
        self.n_fft = 64
        self.hop_length = 16
        self.win_length = 64
        self.window = torch.hann_window(self.win_length)
        
        # Denoising model (2D U-Net)
        # We have num_channels channels, and we represent complex numbers by stacking real/imaginary parts.
        # So input channels = num_channels * 2
        self.denoiser = SpectrogramUNet(in_channels=num_channels * 2, out_channels=num_channels * 2, embed_dim=32, num_classes=2)
        
        # Diffusion schedules
        betas = torch.linspace(beta_start, beta_end, steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def _eeg_to_spectrogram(self, x):
        """Convert [B, C, T] signal to [B, C*2, Freqs, TimeFrames] spectrogram."""
        B, C, T = x.shape
        # Move window buffer to the same device as x
        window = self.window.to(x.device)
        
        # Reshape to perform batch STFT
        x_flat = x.view(B * C, T)
        stft_res = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True
        ) # [B * C, Freqs, TimeFrames]
        
        Freqs, TimeFrames = stft_res.shape[1], stft_res.shape[2]
        stft_res = stft_res.view(B, C, Freqs, TimeFrames)
        
        # Separate real and imaginary parts and stack them along the channel dimension
        real = torch.real(stft_res)
        imag = torch.imag(stft_res)
        spec = torch.cat([real, imag], dim=1) # [B, C*2, Freqs, TimeFrames]
        return spec

    def _spectrogram_to_eeg(self, spec):
        """Convert [B, C*2, Freqs, TimeFrames] spectrogram to [B, C, T] signal."""
        B = spec.shape[0]
        C = spec.shape[1] // 2
        Freqs, TimeFrames = spec.shape[2], spec.shape[3]
        
        real = spec[:, :C]
        imag = spec[:, C:]
        
        # Combine back into a complex tensor
        stft_complex = torch.complex(real, imag)
        stft_complex_flat = stft_complex.view(B * C, Freqs, TimeFrames)
        
        window = self.window.to(spec.device)
        x_flat = torch.istft(
            stft_complex_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=self.seq_len,
            center=True
        ) # [B * C, T]
        
        return x_flat.view(B, C, self.seq_len)

    def q_sample(self, spec_0, t, noise=None):
        """Forward diffusion: add noise to spectrogram."""
        if noise is None:
            noise = torch.randn_like(spec_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * spec_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def training_loss(self, x_0, y):
        """Train standard 2D Spectrogram diffusion model."""
        B = x_0.size(0)
        # Convert to spectrogram
        spec_0 = self._eeg_to_spectrogram(x_0)
        
        # Sample timestep and noise
        t = torch.randint(0, self.steps, (B,), device=x_0.device).long()
        noise = torch.randn_like(spec_0)
        
        # Forward diffuse
        spec_t = self.q_sample(spec_0, t, noise=noise)
        
        # Predict noise
        noise_pred = self.denoiser(spec_t, t.float(), y)
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, spec_t, t, y):
        """Single reverse denoising step on spectrogram."""
        # Predict noise
        t_tensor = torch.tensor([t], device=spec_t.device).float().repeat(spec_t.size(0))
        noise_pred = self.denoiser(spec_t, t_tensor, y)
        
        alpha_t = self.alphas[t]
        beta_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        mean = (1.0 / torch.sqrt(alpha_t)) * (spec_t - (beta_t / sqrt_one_minus_alphas_cumprod_t) * noise_pred)
        
        if t == 0:
            return mean
        else:
            variance = self.posterior_variance[t]
            noise = torch.randn_like(spec_t)
            return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, num_samples, condition_label, device='cpu'):
        """Sample from spectrogram diffusion and invert to 1D EEG."""
        self.eval()
        # Find spectrogram shape by converting a dummy trial
        dummy = torch.zeros(1, self.num_channels, self.seq_len, device=device)
        dummy_spec = self._eeg_to_spectrogram(dummy)
        _, spec_channels, Freqs, TimeFrames = dummy_spec.shape
        
        # Start from noise
        spec = torch.randn(num_samples, spec_channels, Freqs, TimeFrames, device=device)
        y = torch.tensor([condition_label], device=device).repeat(num_samples)
        
        # Reverse denoising
        for t in reversed(range(self.steps)):
            spec = self.p_sample(spec, t, y)
            
        # Convert back to 1D EEG signals
        sampled_signals = self._spectrogram_to_eeg(spec)
        return sampled_signals
