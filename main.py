import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import components
from data.eeg_sim import EEGSimulator
from model.grounded_vae import GroundedVAE
from model.latent_diffusion import LatentDiffusion
from model.baseline_spectrogram import BaselineSpectrogramDiffusion
from evaluation.metrics import evaluate_batch_physiological_metrics, compute_plv_matrix, compute_psd
from validation.classifier import EEGNet, train_classifier, evaluate_classifier

def run_benchmark(epochs_vae=100, epochs_diff=120, num_samples=300, device='cpu'):
    print("=" * 60)
    print("STARTING NEURO-CENTRIC VS 2D SPECTROGRAM BCI BENCHMARK")
    print("=" * 60)
    
    # 1. Initialize simulator and generate dataset
    print("\n[Step 1] Generating Labeled EEG Simulation Dataset...")
    sim = EEGSimulator(sfreq=250, duration=2.0, num_channels=8)
    
    # Generate Motor Imagery dataset
    X_train_raw, y_train = sim.generate_dataset(num_samples=num_samples, task="motor_imagery", noise_level=0.4)
    X_test_raw, y_test = sim.generate_dataset(num_samples=100, task="motor_imagery", noise_level=0.4)
    
    # Normalize datasets to zero mean and unit variance (critical for generative models)
    mean = X_train_raw.mean()
    std = X_train_raw.std()
    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std
    
    print(f"X_train shape: {X_train.shape} (Trials, Channels, Time)")
    print(f"X_test shape: {X_test.shape}")
    print(f"Label distribution - Class 0 (Left Hand): {torch.sum(y_train == 0).item()}, Class 1 (Right Hand): {torch.sum(y_train == 1).item()}")
    
    # 2. Train Grounded VAE
    print(f"\n[Step 2] Training Structured Grounded VAE (epochs={epochs_vae})...")
    vae = GroundedVAE(num_channels=8, seq_len=500, latent_dim=16, sfreq=250).to(device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    
    vae.train()
    for epoch in range(epochs_vae):
        vae_optimizer.zero_grad()
        x_recon, mu, logvar, z = vae(X_train.to(device))
        loss_dict = vae.loss_function(X_train.to(device), x_recon, mu, logvar, z, beta_kl=0.01, beta_ground=10.0)
        loss_dict['loss'].backward()
        vae_optimizer.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs_vae} | Loss: {loss_dict['loss'].item():.4f} | Recon MSE: {loss_dict['recon_loss'].item():.4f} | Grounding Loss: {loss_dict['ground_loss'].item():.4f}")
            
    # 3. Train Native 1D Latent Diffusion Model (LDM)
    print(f"\n[Step 3] Training Native 1D Latent Diffusion Model (epochs={epochs_diff})...")
    ldm = LatentDiffusion(latent_dim=16, steps=100).to(device)
    ldm_optimizer = torch.optim.Adam(ldm.parameters(), lr=0.001)
    
    # Get grounded latent embeddings of training data
    vae.eval()
    with torch.no_grad():
        mu_train, logvar_train = vae.encode(X_train.to(device))
        z_train = vae.reparameterize(mu_train, logvar_train)
        
    ldm.train()
    for epoch in range(epochs_diff):
        ldm_optimizer.zero_grad()
        loss = ldm.training_loss(z_train, y_train.to(device))
        loss.backward()
        ldm_optimizer.step()
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs_diff} | Diffusion Loss: {loss.item():.4f}")
            
    # 4. Train Baseline 2D Spectrogram Diffusion Model
    print(f"\n[Step 4] Training Baseline 2D Spectrogram Diffusion Model (epochs={epochs_diff})...")
    baseline_diff = BaselineSpectrogramDiffusion(num_channels=8, seq_len=500, steps=100).to(device)
    baseline_optimizer = torch.optim.Adam(baseline_diff.parameters(), lr=0.001)
    
    baseline_diff.train()
    for epoch in range(epochs_diff):
        baseline_optimizer.zero_grad()
        loss = baseline_diff.training_loss(X_train.to(device), y_train.to(device))
        loss.backward()
        baseline_optimizer.step()
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:02d}/{epochs_diff} | Baseline Loss: {loss.item():.4f}")
            
    # 5. Generate signals for validation
    print("\n[Step 5] Sampling Generated EEG Trials...")
    num_gen = 50
    
    # Native Neuro-Centric sampling
    gen_neuro_c0, _ = ldm.sample(num_gen, condition_label=0, vae_model=vae, device=device)
    gen_neuro_c1, _ = ldm.sample(num_gen, condition_label=1, vae_model=vae, device=device)
    X_gen_neuro = torch.cat([gen_neuro_c0, gen_neuro_c1], dim=0).cpu()
    y_gen_neuro = torch.cat([torch.zeros(num_gen), torch.ones(num_gen)], dim=0).long()
    
    # Baseline Spectrogram sampling
    gen_base_c0 = baseline_diff.sample(num_gen, condition_label=0, device=device)
    gen_base_c1 = baseline_diff.sample(num_gen, condition_label=1, device=device)
    X_gen_base = torch.cat([gen_base_c0, gen_base_c1], dim=0).cpu()
    y_gen_base = torch.cat([torch.zeros(num_gen), torch.ones(num_gen)], dim=0).long()
    
    # 6. Compute Physiological Validity Metrics
    print("\n[Step 6] Evaluating Biophysical & Physiological Similarity...")
    metrics_neuro = evaluate_batch_physiological_metrics(X_test.numpy(), X_gen_neuro.numpy(), sfreq=250)
    metrics_base = evaluate_batch_physiological_metrics(X_test.numpy(), X_gen_base.numpy(), sfreq=250)
    
    # 7. Evaluate Downstream BCI Classification
    print("\n[Step 7] Running Downstream BCI Classification Validation (EEGNet)...")
    # Train EEGNet on real train dataset
    classifier_real = EEGNet(num_channels=8, seq_len=500, num_classes=2)
    train_classifier(classifier_real, X_train, y_train, epochs=60, lr=0.002, device=device)
    
    # Evaluate classifier on:
    # A. Real Test Data (Target baseline)
    acc_real = evaluate_classifier(classifier_real, X_test, y_test, device=device)
    
    # B. Neuro-Centric Generated Data
    acc_neuro = evaluate_classifier(classifier_real, X_gen_neuro, y_gen_neuro, device=device)
    
    # C. Baseline Spectrogram Generated Data
    acc_base = evaluate_classifier(classifier_real, X_gen_base, y_gen_base, device=device)
    
    # 8. Compile Benchmark Results Table
    print("\n" + "=" * 60)
    print("               BCI BENCHMARK PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"{'Evaluation Metric':<30} | {'Baseline Spectrogram':<20} | {'Native Neuro-Centric':<20}")
    print("-" * 76)
    print(f"{'Phase Coherence PLV (MSE)':<30} | {metrics_base['plv_mse']:.6f}             | {metrics_neuro['plv_mse']:.6f} (Lower is better)")
    print(f"{'PSD Spectrum Shape (MSE)':<30} | {metrics_base['psd_mse']:.6f}             | {metrics_neuro['psd_mse']:.6f} (Lower is better)")
    print(f"{'Phase-Amplitude Coupling Diff':<30} | {metrics_base['pac_diff']:.6f}             | {metrics_neuro['pac_diff']:.6f} (Lower is better)")
    print(f"{'Downstream BCI Acc (EEGNet)':<30} | {acc_base * 100.0:.1f}%                | {acc_neuro * 100.0:.1f}% (Real Test: {acc_real*100.0:.1f}%)")
    print("=" * 76)
    
    # 9. Plotting and Visualizations
    print("\n[Step 8] Creating Visualization Plots...")
    os.makedirs("output", exist_ok=True)
    
    # Plot 1: Compare single channel generated waveforms (Channel 2 - C3)
    plt.figure(figsize=(12, 8))
    times = np.linspace(0, 2.0, 500)
    plt.subplot(3, 1, 1)
    plt.plot(times, X_test[0, 2].numpy(), color='black', alpha=0.8)
    plt.title("Real EEG Trial (Channel C3)")
    plt.ylabel("Voltage (uV)")
    
    plt.subplot(3, 1, 2)
    plt.plot(times, X_gen_neuro[0, 2].numpy(), color='dodgerblue', alpha=0.8)
    plt.title("Native Neuro-Centric Generated Trial (Channel C3)")
    plt.ylabel("Voltage (uV)")
    
    plt.subplot(3, 1, 3)
    plt.plot(times, X_gen_base[0, 2].numpy(), color='crimson', alpha=0.8)
    plt.title("Baseline Spectrogram Generated Trial (Channel C3)")
    plt.ylabel("Voltage (uV)")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("output/generated_signals_comparison.png", dpi=150)
    plt.close()
    
    # Plot 2: Compare Power Spectral Density (PSD)
    f, psd_real = compute_psd(X_test[0].numpy())
    _, psd_neuro = compute_psd(X_gen_neuro[0].numpy())
    _, psd_base = compute_psd(X_gen_base[0].numpy())
    
    plt.figure(figsize=(10, 5))
    plt.plot(f, psd_real[2], color='black', label='Real (C3)', linewidth=2)
    plt.plot(f, psd_neuro[2], color='dodgerblue', label='Neuro-Centric Generated (C3)', linewidth=2)
    plt.plot(f, psd_base[2], color='crimson', label='Baseline Spectrogram Generated (C3)', linewidth=2)
    plt.xlim(1, 50)
    plt.title("Power Spectral Density (PSD) Comparison (1-50 Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Power")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("output/psd_comparison.png", dpi=150)
    plt.close()
    
    # Plot 3: Compare PLV Phase-Coherence Matrix
    plv_real = compute_plv_matrix(X_test[0].numpy())
    plv_neuro = compute_plv_matrix(X_gen_neuro[0].numpy())
    plv_base = compute_plv_matrix(X_gen_base[0].numpy())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ch_names = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    
    im0 = axes[0].imshow(plv_real, vmin=0, vmax=1, cmap='viridis')
    axes[0].set_title("Real PLV")
    axes[0].set_xticks(range(8))
    axes[0].set_xticklabels(ch_names)
    axes[0].set_yticks(range(8))
    axes[0].set_yticklabels(ch_names)
    
    im1 = axes[1].imshow(plv_neuro, vmin=0, vmax=1, cmap='viridis')
    axes[1].set_title("Neuro-Centric PLV")
    axes[1].set_xticks(range(8))
    axes[1].set_xticklabels(ch_names)
    axes[1].set_yticks(range(8))
    axes[1].set_yticklabels(ch_names)
    
    im2 = axes[2].imshow(plv_base, vmin=0, vmax=1, cmap='viridis')
    axes[2].set_title("Baseline Spectrogram PLV")
    axes[2].set_xticks(range(8))
    axes[2].set_xticklabels(ch_names)
    axes[2].set_yticks(range(8))
    axes[2].set_yticklabels(ch_names)
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    
    plt.suptitle("Phase-Coherence PLV Matrices")
    plt.savefig("output/plv_comparison.png", dpi=150)
    plt.close()
    
    print("All plots saved in output/ directory.")
    print("Benchmark complete!")

if __name__ == "__main__":
    run_benchmark(epochs_vae=100, epochs_diff=120, num_samples=200)
