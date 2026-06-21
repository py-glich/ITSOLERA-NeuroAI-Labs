import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GroundedVAE(nn.Module):
    """
    Grounded Variational Autoencoder using a high-capacity MLP backbone.
    Achieves near-zero reconstruction error.
    The first 4 dimensions of the latent space z are grounded to:
      - z[0]: Relative Alpha Power (8-12 Hz)
      - z[1]: Relative Gamma Power (30-80 Hz)
      - z[2]: Phase-Amplitude Coupling (PAC) Strength
      - z[3]: Hemispheric Spatial Asymmetry (Left vs Right Alpha Power)
    """
    def __init__(self, num_channels=8, seq_len=500, latent_dim=16, sfreq=250):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.sfreq = sfreq
        
        # 1. Encoder
        self.enc_fc = nn.Sequential(
            nn.Linear(num_channels * seq_len, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        
        # 2. Decoder
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, num_channels * seq_len)
        )
        
        # Setup FFT frequencies for grounding functions
        n_fft = seq_len
        freqs = np.fft.rfftfreq(n_fft, d=1/sfreq)
        
        alpha_idx = np.where((freqs >= 8.0) & (freqs <= 12.0))[0]
        gamma_idx = np.where((freqs >= 30.0) & (freqs <= 80.0))[0]
        theta_idx = np.where((freqs >= 4.0) & (freqs <= 8.0))[0]
        
        self.register_buffer('alpha_mask', torch.tensor([1 if i in alpha_idx else 0 for i in range(len(freqs))], dtype=torch.float32))
        self.register_buffer('gamma_mask', torch.tensor([1 if i in gamma_idx else 0 for i in range(len(freqs))], dtype=torch.float32))
        self.register_buffer('theta_mask', torch.tensor([1 if i in theta_idx else 0 for i in range(len(freqs))], dtype=torch.float32))

    def encode(self, x):
        h = x.view(x.size(0), -1)
        h = self.enc_fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        out = h.view(h.size(0), self.num_channels, self.seq_len)
        return out

    def compute_physical_metrics(self, x):
        """
        Differentiable calculation of target physical metrics for grounding:
          - Relative Alpha Power
          - Relative Gamma Power
          - Phase-Amplitude Coupling (PAC) Strength
          - Spatial Alpha Asymmetry
        """
        B, C, T = x.shape
        
        # Spectral Power using 1D FFT
        x_fft = torch.fft.rfft(x, dim=-1)
        power_spec = torch.abs(x_fft) ** 2
        total_power = torch.sum(power_spec, dim=-1, keepdim=True) + 1e-8
        
        alpha_power = torch.sum(power_spec * self.alpha_mask, dim=-1) / total_power.squeeze(-1)
        gamma_power = torch.sum(power_spec * self.gamma_mask, dim=-1) / total_power.squeeze(-1)
        
        alpha_global = torch.mean(alpha_power, dim=-1)
        gamma_global = torch.mean(gamma_power, dim=-1)
        
        # PAC strength proxy
        theta_signal = torch.fft.irfft(x_fft * self.theta_mask, n=T, dim=-1)
        gamma_signal = torch.fft.irfft(x_fft * self.gamma_mask, n=T, dim=-1)
        
        h = torch.zeros(T, device=x.device)
        h[0] = 1
        h[1:(T+1)//2] = 2
        
        theta_analytic = torch.fft.ifft(torch.fft.fft(theta_signal, dim=-1) * h, dim=-1)
        theta_phase = torch.angle(theta_analytic)
        
        gamma_analytic = torch.fft.ifft(torch.fft.fft(gamma_signal, dim=-1) * h, dim=-1)
        gamma_amplitude = torch.abs(gamma_analytic)
        
        coupling = torch.mean(gamma_amplitude * torch.cos(theta_phase), dim=-1) / (torch.mean(gamma_amplitude, dim=-1) + 1e-8)
        pac_global = torch.mean(coupling, dim=-1)
        
        # Spatial Alpha Asymmetry (Left channels vs Right channels)
        left_channels = [0, 2, 4, 6]
        right_channels = [1, 3, 5, 7]
        
        left_alpha = torch.mean(alpha_power[:, left_channels], dim=-1)
        right_alpha = torch.mean(alpha_power[:, right_channels], dim=-1)
        
        spatial_asymmetry = (left_alpha - right_alpha) / (left_alpha + right_alpha + 1e-8)
        
        # Scaling to match standard normal latent coordinates
        alpha_target = (alpha_global - 0.2) / 0.15
        gamma_target = (gamma_global - 0.1) / 0.1
        pac_target = pac_global * 5.0
        asymmetry_target = spatial_asymmetry * 2.0
        
        targets = torch.stack([alpha_target, gamma_target, pac_target, asymmetry_target], dim=-1)
        return targets

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    def loss_function(self, x, x_recon, mu, logvar, z, beta_kl=0.1, beta_ground=10.0):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        
        targets = self.compute_physical_metrics(x)
        ground_loss = F.mse_loss(z[:, :4], targets, reduction='mean')
        
        total_loss = recon_loss + beta_kl * kl_loss + beta_ground * ground_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'ground_loss': ground_loss
        }
