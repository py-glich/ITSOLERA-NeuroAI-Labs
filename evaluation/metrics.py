import numpy as np
import scipy.signal as signal

def compute_analytical_phase(x, sfreq=250):
    """
    Extract instantaneous phase using the Hilbert transform.
    x shape: [num_channels, time_samples] or [time_samples]
    """
    # If 1D, make 2D
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x[None, :]
        
    num_channels, n = x.shape
    analytical = signal.hilbert(x, axis=-1)
    phase = np.angle(analytical)
    
    if is_1d:
        return phase[0]
    return phase


def compute_plv_matrix(signals, sfreq=250):
    """
    Compute pairwise Phase Locking Value (PLV) matrix between all channels.
    signals shape: [num_channels, time_samples]
    PLV_ij = |1/T * sum_t exp(i * (phase_i(t) - phase_j(t)))|
    """
    num_channels, n = signals.shape
    phases = compute_analytical_phase(signals, sfreq)
    
    plv_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        for j in range(num_channels):
            phase_diff = phases[i] - phases[j]
            plv_matrix[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
            
    return plv_matrix


def compute_psd(signals, sfreq=250):
    """
    Compute normalized Power Spectral Density (PSD) for each channel using Welch's method.
    signals shape: [num_channels, time_samples]
    Returns freqs, psd_normalized [num_channels, freqs]
    """
    num_channels, n = signals.shape
    psds = []
    freqs = None
    
    for ch in range(num_channels):
        f, p = signal.welch(signals[ch], fs=sfreq, nperseg=min(n, 128))
        psds.append(p)
        if freqs is None:
            freqs = f
            
    psds = np.stack(psds, axis=0)
    # Normalize each channel's PSD to sum to 1 to represent relative distribution
    psd_norm = psds / (np.sum(psds, axis=-1, keepdims=True) + 1e-8)
    return freqs, psd_norm


def compute_pac_index(sig, low_freq=(4, 8), high_freq=(30, 80), sfreq=250):
    """
    Compute Phase-Amplitude Coupling (PAC) proxy using the Modulation Index style.
    sig shape: [time_samples] (typically a channel like O1)
    """
    n = len(sig)
    # Bandpass filter for low frequency (theta)
    nyq = 0.5 * sfreq
    b_low, a_low = signal.butter(4, [low_freq[0]/nyq, low_freq[1]/nyq], btype='band')
    theta_sig = signal.lfilter(b_low, a_low, sig)
    
    # Bandpass filter for high frequency (gamma)
    b_high, a_high = signal.butter(4, [high_freq[0]/nyq, high_freq[1]/nyq], btype='band')
    gamma_sig = signal.lfilter(b_high, a_high, sig)
    
    # Extract phase of theta and amplitude of gamma
    theta_phase = np.angle(signal.hilbert(theta_sig))
    gamma_amp = np.abs(signal.hilbert(gamma_sig))
    
    # Coupling index: mean(A_gamma * cos(theta_phase)) normalized by mean(A_gamma)
    pac_val = np.mean(gamma_amp * np.cos(theta_phase)) / (np.mean(gamma_amp) + 1e-8)
    return np.abs(pac_val)


def evaluate_batch_physiological_metrics(real_batch, gen_batch, sfreq=250):
    """
    Evaluate physiological similarity metrics over a batch of EEG trials.
    real_batch, gen_batch: [B, C, T]
    Returns a dictionary of mean metrics.
    """
    B, C, T = real_batch.shape
    
    plv_mses = []
    psd_mses = []
    pac_diffs = []
    
    for i in range(B):
        # 1. PLV Matrix comparison
        plv_real = compute_plv_matrix(real_batch[i], sfreq)
        plv_gen = compute_plv_matrix(gen_batch[i], sfreq)
        plv_mses.append(np.mean((plv_real - plv_gen) ** 2))
        
        # 2. PSD Similarity comparison
        _, psd_real = compute_psd(real_batch[i], sfreq)
        _, psd_gen = compute_psd(gen_batch[i], sfreq)
        psd_mses.append(np.mean((psd_real - psd_gen) ** 2))
        
        # 3. PAC Index comparison (evaluating Channel 6 - O1 which has synthetic PAC)
        pac_real = compute_pac_index(real_batch[i, 6], sfreq=sfreq)
        pac_gen = compute_pac_index(gen_batch[i, 6], sfreq=sfreq)
        pac_diffs.append(np.abs(pac_real - pac_gen))
        
    return {
        'plv_mse': np.mean(plv_mses),
        'psd_mse': np.mean(psd_mses),
        'pac_diff': np.mean(pac_diffs)
    }
