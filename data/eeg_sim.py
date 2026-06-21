import numpy as np
import torch
from scipy.signal import butter, lfilter

class EEGSimulator:
    def __init__(self, sfreq=250, duration=2.0, num_channels=8):
        self.sfreq = sfreq
        self.duration = duration
        self.num_channels = num_channels
        self.num_samples = int(sfreq * duration)
        self.times = np.linspace(0, duration, self.num_samples)
        
        # 10-20 system electrode locations (approximate 3D coordinates on unit sphere)
        # F3, F4, C3, C4, P3, P4, O1, O2
        self.electrode_names = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
        self.electrode_coords = np.array([
            [-0.5,  0.5,  0.7],  # F3
            [ 0.5,  0.5,  0.7],  # F4
            [-0.7,  0.0,  0.7],  # C3
            [ 0.7,  0.0,  0.7],  # C4
            [-0.5, -0.5,  0.7],  # P3
            [ 0.5, -0.5,  0.7],  # P4
            [-0.3, -0.8,  0.5],  # O1
            [ 0.3, -0.8,  0.5],  # O2
        ])
        
    def get_distance_matrix(self):
        """Compute pairwise Euclidean distance matrix between electrodes."""
        diff = self.electrode_coords[:, None, :] - self.electrode_coords[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1))

    def _generate_pink_noise(self, size):
        """Generate 1/f pink noise using spectral integration."""
        unequal_size = size + (size % 2)
        white = np.random.randn(unequal_size)
        fft_white = np.fft.rfft(white)
        # Scale frequencies by 1 / sqrt(f) to get 1/f power spectral density
        freqs = np.fft.rfftfreq(unequal_size)
        freqs[0] = freqs[1]  # avoid division by zero
        fft_pink = fft_white / np.sqrt(freqs)
        pink = np.fft.irfft(fft_pink)
        return pink[:size]

    def _apply_bandpass_filter(self, data, lowcut, highcut, order=4):
        """Apply a Butterworth bandpass filter."""
        nyq = 0.5 * self.sfreq
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def generate_trial(self, condition="motor_left", noise_level=0.5):
        """
        Generates a single trial of 8-channel EEG.
        
        Conditions:
          - 'motor_left': Left hand motor imagery -> alpha (8-12Hz) suppression in right hemisphere (C4, P4, O2)
          - 'motor_right': Right hand motor imagery -> alpha (8-12Hz) suppression in left hemisphere (C3, P3, O1)
          - 'epileptic': Contains focal epileptic spikes centered at O1, propagating to nearby electrodes.
          - 'normal': Baseline background state with standard oscillations.
        """
        # 1. Generate underlying source activities
        # We simulate 4 independent deep sources: Left Motor, Right Motor, Occipital, and Frontal
        source_coords = np.array([
            [-0.6,  0.1,  0.4],  # Left motor source (near C3)
            [ 0.6,  0.1,  0.4],  # Right motor source (near C4)
            [ 0.0, -0.7,  0.2],  # Occipital source (near O1/O2)
            [ 0.0,  0.6,  0.3],  # Frontal source (near F3/F4)
        ])
        
        num_sources = len(source_coords)
        source_signals = np.zeros((num_sources, self.num_samples))
        
        # Pink noise base for each source
        for i in range(num_sources):
            source_signals[i] = self._generate_pink_noise(self.num_samples) * 0.3
            
        # Add basic rhythms (theta, alpha, beta, gamma)
        # Theta: 6 Hz
        theta = np.sin(2 * np.pi * 6 * self.times)
        # Alpha: 10 Hz
        alpha = np.sin(2 * np.pi * 10 * self.times)
        # Beta: 20 Hz
        beta = np.sin(2 * np.pi * 20 * self.times)
        
        # Source 0 (Left Motor): strong alpha, unless right motor imagery is occurring
        alpha_l = alpha.copy()
        if condition == "motor_right":
            alpha_l *= 0.2  # Desynchronization / Suppression
        source_signals[0] += 0.5 * alpha_l + 0.2 * beta
        
        # Source 1 (Right Motor): strong alpha, unless left motor imagery is occurring
        alpha_r = alpha.copy()
        if condition == "motor_left":
            alpha_r *= 0.2  # Desynchronization / Suppression
        source_signals[1] += 0.5 * alpha_r + 0.2 * beta
        
        # Source 2 (Occipital): strong alpha, and Phase-Amplitude Coupling (PAC)
        # Theta phase modulates Gamma amplitude (40 Hz)
        theta_carrier = np.sin(2 * np.pi * 5 * self.times) # 5Hz theta
        gamma_amp = 1.0 + 0.8 * theta_carrier  # Amplitude envelope modulated by theta
        gamma_signal = gamma_amp * np.sin(2 * np.pi * 45 * self.times)  # 45Hz gamma
        source_signals[2] += 0.4 * alpha + 0.3 * gamma_signal
        
        # Source 3 (Frontal): strong theta activity (drowsiness / control)
        source_signals[3] += 0.6 * theta
        
        # Add epileptic discharges for epileptic condition in occipital source
        if condition == "epileptic":
            # Generate spike-and-wave discharges
            # 3 spike events in 2 seconds
            spike_times = [0.4, 1.0, 1.6]
            for st in spike_times:
                idx = int(st * self.sfreq)
                # Spike: sharp transient
                t_spike = np.linspace(-0.05, 0.05, int(0.1 * self.sfreq))
                spike = 4.0 * np.exp(-150 * (t_spike)**2) * np.sin(2 * np.pi * 15 * t_spike)
                # Wave: slow wave following spike
                t_wave = np.linspace(0, 0.3, int(0.3 * self.sfreq))
                wave = -2.0 * np.sin(2 * np.pi * 2.5 * t_wave) * np.exp(-8 * t_wave)
                
                # Apply to occipital source
                source_signals[2, idx:idx+len(spike)] += spike
                source_signals[2, idx+len(spike):idx+len(spike)+len(wave)] += wave

        # 2. Project sources to electrodes (Spatial Volume Conduction)
        # Project using electric potential distance decay: V = S * exp(-d / lambda)
        # Where d is distance between source and electrode, lambda = 0.5
        lambda_decay = 0.5
        scalp_signals = np.zeros((self.num_channels, self.num_samples))
        for ch in range(self.num_channels):
            for src in range(num_sources):
                d = np.sqrt(np.sum((self.electrode_coords[ch] - source_coords[src])**2))
                weight = np.exp(-d / lambda_decay)
                scalp_signals[ch] += weight * source_signals[src]

        # 3. Add independent scalp sensor noise (pink noise + white noise)
        for ch in range(self.num_channels):
            sensor_noise = (self._generate_pink_noise(self.num_samples) * 0.7 + 
                            np.random.randn(self.num_samples) * 0.3)
            scalp_signals[ch] += noise_level * sensor_noise
            
        return scalp_signals

    def generate_dataset(self, num_samples=200, task="motor_imagery", noise_level=0.5):
        """
        Generates a labeled PyTorch dataset.
        If task == 'motor_imagery':
            class 0: motor_left, class 1: motor_right
        If task == 'epilepsy':
            class 0: normal, class 1: epileptic
        """
        X, y = [], []
        
        if task == "motor_imagery":
            conditions = ["motor_left", "motor_right"]
        elif task == "epilepsy":
            conditions = ["normal", "epileptic"]
        else:
            raise ValueError(f"Unknown task {task}")
            
        for i in range(num_samples):
            label = np.random.randint(0, 2)
            cond = conditions[label]
            signal = self.generate_trial(condition=cond, noise_level=noise_level)
            X.append(signal)
            y.append(label)
            
        X = np.stack(X, axis=0) # [num_samples, channels, times]
        y = np.array(y)
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

if __name__ == "__main__":
    # Test simulator
    sim = EEGSimulator()
    X, y = sim.generate_dataset(num_samples=10, task="motor_imagery")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Distance Matrix:\n", sim.get_distance_matrix()[:3, :3])
