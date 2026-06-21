# Native Neuro-Centric Generative Architectures for Brain-Computer Interfaces

This implementation plan outlines the development of a native 1D Generative Diffusion Transformer for electroencephalogram (EEG) signals that addresses the architectural mismatch of using 2D computer vision models. The project will implement biophysically constrained attention layers, a grounded and interpretable latent space, and comparative benchmarking against 2D spectrogram-based diffusion models.

## User Review Required

- **Execution Environment**: Standard sandboxed command execution is currently encountering a system-level permission error (`opening NUL for ACL write: Access is denied`). To compile and run the evaluation script, we will request a single sandbox bypass (`BypassSandbox: true`).
- **No Placeholders**: All components will be implemented using PyTorch, NumPy, SciPy, and Scikit-Learn without placeholders, producing a fully functioning demonstration, benchmark tables, and visualization outputs.

## Open Questions

- **Electrode Layout**: We will simulate an 8-channel EEG layout corresponding to a standard subset of the 10-20 international system (e.g., F3, F4, C3, C4, P3, P4, O1, O2). Is this standard layout acceptable for modeling spatial volume conduction?
- **Sampling Rate**: We will use a standard sampling rate of 250 Hz with 2-second windows (500 time steps), which is standard for clinical and consumer BCI systems.

---

## Proposed Changes

We will create a new project directory `neuro_centric_bci` under `C:\Users\kk\.gemini\antigravity\scratch\`.

```
neuro_centric_bci/
├── requirements.txt
├── main.py
├── technical_report.md
├── data/
│   └── eeg_sim.py
├── model/
│   ├── biophysical_priors.py
│   ├── grounded_vae.py
│   ├── latent_diffusion.py
│   └── baseline_spectrogram.py
├── evaluation/
│   └── metrics.py
└── validation/
    └── classifier.py
```

### 1. Biophysical Prior Constraints in Attention

#### [NEW] [biophysical_priors.py](file:///C:/Users/kk/.gemini/antigravity/scratch/neuro_centric_bci/model/biophysical_priors.py)
We will implement three key biophysical priors within the 1D Transformer's multi-head attention:
- **Frequency Band Hierarchy**: Restricts attention heads to specific frequency bands (Delta, Theta, Alpha, Beta, Gamma) using bandpass filter masking of Query-Key comparisons.
- **Spatial Volume Conduction Fields**: Modulates channel-to-channel attention matrix $A$ by scalp distance $D_{ij}$ between electrodes: $A'_{ij} = A_{ij} \cdot e^{-\lambda D_{ij}}$, modeling the physical volume conduction of electrical fields.
- **Phase-Amplitude Coupling (PAC) Modulation**: Implements cross-frequency coupling where high-frequency (gamma) representations are modulated/gated by low-frequency (theta) phase features.

### 2. Structured Latent Space Grounding

#### [NEW] [grounded_vae.py](file:///C:/Users/kk/.gemini/antigravity/scratch/neuro_centric_bci/model/grounded_vae.py)
We will implement a Variational Autoencoder (VAE) that projects 1D multichannel EEG signals into a structured latent space $z \in \mathbb{R}^d$. We will ground the first 4 dimensions of the latent space to correspond to specific neurophysiological constructs:
- $z_0$: Alpha relative power (8-12 Hz) - related to relaxation/visual processing.
- $z_1$: Gamma relative power (30-80 Hz) - related to active cognitive processing.
- $z_2$: Phase-Amplitude Coupling (PAC) index - coupling strength between theta phase and gamma amplitude.
- $z_3$: Spatial asymmetry - hemispheric differences (e.g., C3 vs C4 representing motor planning).

We will enforce this grounding by adding a **grounding loss** $\mathcal{L}_{\text{ground}}$:
\[ \mathcal{L}_{\text{ground}} = \sum_{k=0}^{3} \text{MSE}(z_k, \phi_k(x)) \]
where $\phi_k(x)$ is a deterministic signal processing function (e.g., FFT or Hilbert transform) that computes the physical metric directly from the input signal $x$. The remaining latent dimensions will capture residual unconstrained variations.

### 3. Native 1D Generative Diffusion Transformer

#### [NEW] [latent_diffusion.py](file:///C:/Users/kk/.gemini/antigravity/scratch/neuro_centric_bci/model/latent_diffusion.py)
A 1D Diffusion Transformer (DiT) will operate in the grounded latent space of our VAE:
- It will perform forward and reverse diffusion processes in latent space.
- It will be conditioned on the diffusion step $t$ and BCI conditions (e.g., Left vs Right motor imagery, or Normal vs Epileptic discharge).
- Once trained, it will sample new latent vectors that can be decoded by the VAE decoder into realistic, multi-channel EEG signals.
- This allows controlled generation: e.g., we can manually alter the grounded latent dimensions (like increasing alpha suppression) and decode the resulting signal.

### 4. Baseline Model (2D Spectrogram Diffusion)

#### [NEW] [baseline_spectrogram.py](file:///C:/Users/kk/.gemini/antigravity/scratch/neuro_centric_bci/model/baseline_spectrogram.py)
To prove the superiority of native 1D modeling, we will implement a standard 2D Spectrogram-based Diffusion Model:
- Converts 1D EEG signals to 2D Spectrograms via Short-Time Fourier Transform (STFT).
- Applies a standard 2D Diffusion model (U-Net) to generate/denoise the spectrogram.
- Converts back to 1D via Inverse STFT (iSTFT).
- This baseline will highlight the issues of phase destruction and volume conduction neglect.

### 5. Data Generation

#### [NEW] [eeg_sim.py](file:///C:/Users/kk/.gemini/antigravity/scratch/neuro_centric_bci/data/eeg_sim.py)
Generates biophysically realistic synthetic EEG signals for training and evaluation. It will model:
- Multiple standard channels (e.g., 8 channels).
- Motor Imagery class: Left-hand imagery induces contralateral alpha suppression (C4 desynchronization) vs Right-hand imagery (C3 desynchronization).
- Epileptic Discharge class: Occasional high-amplitude spikes across channels with specific spatial propagation.
- Realistic volume conduction and noise (pink noise / $1/f$ noise).

### 6. Physiological Validation & Downstream Classifier

#### [NEW] [metrics.py](file:///C:/Users/kk/.gemini/antigravity/scratch/neuro_centric_bci/evaluation/metrics.py)
Implements physiological validity metrics:
- **Phase Locking Value (PLV)**: To assess phase coherence between channels.
- **Power Spectral Density (PSD) Similarity**: Mean squared error between generated and real power spectra.
- **Phase-Amplitude Coupling (PAC) Index**: To assess cross-frequency modulation.
- **Reconstruction Fidelity**: Signal-to-Noise Ratio (SNR) or Mean Squared Error (MSE) on denoising tasks.

#### [NEW] [classifier.py](file:///C:/Users/kk/.gemini/antigravity/scratch/neuro_centric_bci/validation/classifier.py)
An EEGNet-style deep classifier that takes 1D EEG signals and classifies the condition (Motor Imagery Left/Right or Epileptic Normal/Abnormal). We will evaluate:
- Classification accuracy on real test data when the classifier is trained on **real data**.
- Classification accuracy when trained on **generated data** from our Neuro-Centric model vs the Baseline model.
- Classification accuracy when evaluating **denoised signals** from both models.

---

## Verification Plan

### Automated Tests
We will write a comprehensive test suite in `main.py` which will:
1. Generate synthetic EEG datasets (motor imagery and epileptic discharges).
2. Train the VAE and Latent Diffusion Model.
3. Train the baseline Spectrogram Diffusion Model.
4. Run denoising tests on both models.
5. Compute physiological metrics (PLV, PSD, PAC) for both models.
6. Train the downstream classifier and verify classification accuracy.
7. Print a comparison table comparing the Native Neuro-Centric model against the baseline.

We will run the verification command:
```powershell
python main.py --run_benchmark
```
We will request standard sandbox bypass to execute this script and output the results.

### Manual Verification
- We will inspect the generated plots (saved in `neuro_centric_bci/output/`) illustrating the reconstructed signals, the latent space trajectories, and comparative power spectra.
- We will document all findings, mathematical formulations, and results in `technical_report.md` and `walkthrough.md`.
