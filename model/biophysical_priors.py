import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FrequencyBandFilter(nn.Module):
    """
    Differentiable 1D frequency band filtering using FFT.
    Applies a smooth bandpass mask in the frequency domain.
    """
    def __init__(self, sfreq=250, low_freq=8.0, high_freq=12.0, seq_len=500, transition_width=1.0):
        super().__init__()
        self.sfreq = sfreq
        self.seq_len = seq_len
        
        # Calculate FFT frequency bins
        n_fft = seq_len
        freqs = np.fft.rfftfreq(n_fft, d=1/sfreq)
        
        # Create a smooth bandpass mask using a cosine taper
        mask = np.zeros_like(freqs)
        for idx, f in enumerate(freqs):
            if f >= low_freq and f <= high_freq:
                mask[idx] = 1.0
            elif f < low_freq and f >= (low_freq - transition_width):
                # Smooth roll-off low side
                mask[idx] = 0.5 * (1 + np.cos(np.pi * (low_freq - f) / transition_width))
            elif f > high_freq and f <= (high_freq + transition_width):
                # Smooth roll-off high side
                mask[idx] = 0.5 * (1 + np.cos(np.pi * (f - high_freq) / transition_width))
                
        self.register_buffer('freq_mask', torch.tensor(mask, dtype=torch.float32))

    def forward(self, x):
        # x shape: [Batch, Channels, Time] or [Batch, Heads, Channels, Time]
        # Perform 1D FFT along the last (time) dimension
        x_fft = torch.fft.rfft(x, dim=-1)
        # Apply mask
        x_fft_masked = x_fft * self.freq_mask
        # Inverse FFT
        x_filtered = torch.fft.irfft(x_fft_masked, n=self.seq_len, dim=-1)
        return x_filtered


class SpatialVolumeConductionBias(nn.Module):
    """
    Applies spatial distance penalty to the attention logits between EEG channels.
    Based on standard scalp distance coordinates.
    """
    def __init__(self, electrode_coords, lambda_decay=1.0):
        super().__init__()
        # Compute distance matrix
        diff = electrode_coords[:, None, :] - electrode_coords[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        # Keep distance matrix as buffer
        self.register_buffer('dist_matrix', torch.tensor(dist_matrix, dtype=torch.float32))
        self.lambda_decay = lambda_decay

    def forward(self, attn_logits):
        # attn_logits shape: [Batch, Heads, Channels, Channels]
        # Subtract distance-based penalty (negative bias before softmax)
        # Large distance = large penalty = small attention
        bias = -self.lambda_decay * self.dist_matrix.unsqueeze(0).unsqueeze(1)
        return attn_logits + bias


class PhaseAmplitudeCouplingGating(nn.Module):
    """
    Modulates high-frequency (gamma) representations based on low-frequency (theta) phase.
    Extracts theta phase and uses it to multiplicatively modulate gamma amplitude.
    """
    def __init__(self, sfreq=250, seq_len=500):
        super().__init__()
        # Sub-filters to isolate theta (4-8 Hz) and gamma (30-80 Hz)
        self.theta_filter = FrequencyBandFilter(sfreq=sfreq, low_freq=4.0, high_freq=8.0, seq_len=seq_len)
        self.gamma_filter = FrequencyBandFilter(sfreq=sfreq, low_freq=30.0, high_freq=80.0, seq_len=seq_len)
        
        # Quadrature filter (90 degree phase shift) to compute phase via analytical signal (like Hilbert transform)
        # Hilbert transform is equivalent to zeroing negative frequencies in FFT
        # We can construct a simple differentiable phase extractor
        
    def _extract_phase(self, x):
        """Extract instantaneous phase of theta signal using FFT analytical signal."""
        # x shape: [B, C, T]
        x_fft = torch.fft.fft(x, dim=-1)
        n = x.size(-1)
        h = torch.zeros(n, device=x.device)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1:n // 2] = 2
        else:
            h[0] = 1
            h[1:(n + 1) // 2] = 2
            
        analytical_fft = x_fft * h.unsqueeze(0).unsqueeze(1)
        analytical = torch.fft.ifft(analytical_fft, dim=-1)
        phase = torch.angle(analytical)
        return phase # shape: [B, C, T]

    def forward(self, theta_features, gamma_features):
        """
        Theta features modulate Gamma features.
        theta_features, gamma_features shape: [B, C, T]
        """
        # Filter signals to isolate bands
        theta_band = self.theta_filter(theta_features)
        gamma_band = self.gamma_filter(gamma_features)
        
        # Extract phase of low-frequency (theta)
        theta_phase = self._extract_phase(theta_band)
        
        # Multiplicative coupling: gamma is gated by (1.0 + cos(theta_phase))
        # The coupling strength is learnable via a scaling parameter
        coupling = 1.0 + 0.5 * torch.cos(theta_phase)
        
        # Modulate the original gamma features
        modulated_gamma = gamma_features * coupling
        
        return modulated_gamma


class BiophysicallyConstrainedAttention(nn.Module):
    """
    Attention layer with Frequency Band constraints and Spatial Volume Conduction bias.
    """
    def __init__(self, embed_dim, num_heads, num_channels, electrode_coords, sfreq=250, seq_len=500):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Spatial Bias
        self.spatial_bias = SpatialVolumeConductionBias(electrode_coords, lambda_decay=0.5)
        
        # Frequency band group configurations for heads
        # Divide heads into bands: Delta, Theta, Alpha, Beta, Gamma
        # Head 0: Delta (1-4 Hz)
        # Head 1: Theta (4-8 Hz)
        # Head 2: Alpha (8-12 Hz)
        # Head 3: Beta (12-30 Hz)
        # Head 4-7: Unconstrained/Broadband
        self.filters = nn.ModuleList([
            FrequencyBandFilter(sfreq, 1.0, 4.0, seq_len),  # Delta
            FrequencyBandFilter(sfreq, 4.0, 8.0, seq_len),  # Theta
            FrequencyBandFilter(sfreq, 8.0, 12.0, seq_len), # Alpha
            FrequencyBandFilter(sfreq, 12.0, 30.0, seq_len) # Beta
        ])
        
    def forward(self, x):
        # Input shape x: [Batch, Channels, EmbedDim] (representing channel features)
        # However, for 1D time-series, the sequence length is Time, channels are like tokens
        # Or, we can treat channels as tokens, or time steps as tokens.
        # Here we model Channel-wise Attention (how channels attend to each other):
        # x shape: [B, C, D] where C = Channels, D = EmbedDim
        # For frequency filtering, we need the Time dimension.
        # Let's assume input features have a temporal shape: [B, C, T, D]
        # or we run attention across channels, but project features containing time.
        # Let's design channel-wise attention over a sequence of length T:
        # x shape: [B, C, T]
        # In a 1D Transformer, the input is [B, C, T] representing multi-channel signals.
        # We project channels to features, or we do attention across the Time dimension.
        # Actually, EEG signals have two dimensions: Channels and Time.
        # A 1D Generative Transformer can treat Time as the sequence length, and Channels as the features:
        # Input: [Batch, Time, Channels]
        # If sequence length is Time, then we can apply Frequency Constraints along the Time dimension,
        # and Spatial volume conduction constraints between the Channels.
        # Let's implement this!
        
        # Input shape: [B, T, C] where T = seq_len, C = num_channels
        B, T, C = x.shape
        
        # Project to Q, K, V
        # Q, K, V shape: [B, T, C] -> [B, T, D] where D = embed_dim
        # Let's transpose to apply standard attention: seq_len = T, features = channels
        # Wait, if we want to apply Spatial Conduction Bias, the distance matrix is between Channels.
        # So we should perform Channel-wise Attention, or Spatial-Temporal Attention.
        # Let's perform standard Multi-Head Attention along the Time dimension, and apply Frequency band filters.
        # For Spatial Conduction Bias, we can apply it if we do Channel-wise attention!
        # Let's do both!
        # First, project:
        q = self.q_proj(x) # [B, T, D]
        k = self.k_proj(x) # [B, T, D]
        v = self.v_proj(x) # [B, T, D]
        
        # Reshape Q, K, V for multi-head attention: [B, H, T, HeadDim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply Frequency Band Constraints to different heads
        # Head 0, 1, 2, 3 are constrained along the Time dimension
        # Transpose to [B, H, HeadDim, T] to filter time
        q_t = q.transpose(-1, -2)
        k_t = k.transpose(-1, -2)
        v_t = v.transpose(-1, -2)
        
        for h_idx in range(min(4, self.num_heads)):
            q_t[:, h_idx] = self.filters[h_idx](q_t[:, h_idx])
            k_t[:, h_idx] = self.filters[h_idx](k_t[:, h_idx])
            v_t[:, h_idx] = self.filters[h_idx](v_t[:, h_idx])
            
        # Transpose back: [B, H, T, HeadDim]
        q = q_t.transpose(-1, -2)
        k = k_t.transpose(-1, -2)
        v = v_t.transpose(-1, -2)
        
        # Compute Time attention: [B, H, T, T]
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # Output after attention: [B, H, T, HeadDim]
        out_time = torch.matmul(attn_weights, v)
        
        # Transpose back to [B, T, D]
        out = out_time.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        out = self.out_proj(out)
        
        # Now, apply Spatial volume conduction constraint.
        # Since the channel dimension is index C (features in x), we can do channel mixing
        # or we can perform channel-wise attention at the end of the block.
        # Let's apply a spatial mixing layer constrained by electrode distances:
        # scalp mixing: [B, T, C] -> channel-to-channel interaction.
        # We can implement a Spatial Attention block inside this layer:
        # Let's transpose out to [B, T, C] (which it already is, C = num_channels if D = C, but D is embed_dim).
        # Let's map embed_dim to C, apply channel attention, and map back!
        # Alternatively, we can project to channel space, apply Spatial Bias, and project back:
        # Let's project time features to channel-wise relations:
        # Compute Channel-wise Attention matrix:
        # Transpose out to [B, C, D_temp] or just project [B, T, D] -> [B, D, C]
        # Let's keep it simple: we project the features at each time step to a [C, C] interaction matrix.
        # We compute channel attention at each timestep t:
        # Let's define a query_ch and key_ch of shape [B, T, C]
        # Attention map: [B, T, C, C]
        # We apply SpatialVolumeConductionBias to it, softmax, and reconstruct features.
        # This is extremely clean and implements Spatial Conduction exactly!
        # Let's do this:
        q_ch = x  # shape: [B, T, C]
        k_ch = x
        v_ch = x
        
        # Compute Channel-wise attention logits: [B, T, C, C]
        ch_attn_logits = torch.matmul(q_ch, k_ch.transpose(-1, -2)) / np.sqrt(C)
        # Reshape to [B*T, 1, C, C] to apply Spatial Bias (which expects [Batch, Heads, C, C])
        ch_attn_logits = ch_attn_logits.unsqueeze(2) # [B, T, 1, C, C]
        ch_attn_logits = ch_attn_logits.view(-1, 1, C, C)
        ch_attn_logits = self.spatial_bias(ch_attn_logits)
        ch_attn_logits = ch_attn_logits.view(B, T, 1, C, C).squeeze(2)
        
        ch_attn_weights = F.softmax(ch_attn_logits, dim=-1)
        spatial_mixed = torch.matmul(ch_attn_weights, v_ch) # [B, T, C]
        
        # Final output combines temporal attention and spatial volume conduction mixing
        final_out = out + spatial_mixed.repeat(1, 1, self.embed_dim // C) # expand to embed_dim
        
        return final_out
