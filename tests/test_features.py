#!/usr/bin/env python3
"""
Test script to load mixture_4ch.wav, compute STFT, and generate SRP-PHAT map with peak detection.
"""

import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirokai_doa.utils import load_wav_best_effort
from mirokai_doa.features import stft_multi, compute_mag_phase_cos_sin
from mirokai_doa.mic import Params, ring_geometry_meters
from mirokai_doa.srp import estimate_azimuths_360_from_stft, srp_weighted_VAD_phat_azimuth_map_360_from_stft
from mirokai_doa.vad import VoiceActivityDetector


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_mic_config_from_yaml(config):
    """Extract microphone configuration from YAML config."""
    mics = config['array']['mics']
    mic_positions = []
    for mic in mics:
        # Convert mm to meters
        x_m = mic['x_mm'] / 1000.0
        y_m = mic['y_mm'] / 1000.0
        mic_positions.append([x_m, y_m])
    return np.array(mic_positions)


def get_stft_params_from_yaml(config):
    """Extract STFT parameters from YAML config."""
    stft_config = config['stft']
    return {
        'win_s': stft_config['window_ms'] / 1000.0,  # Convert ms to seconds
        'hop_s': stft_config['hop_ms'] / 1000.0,     # Convert ms to seconds
        'nfft': stft_config['fft_size'],
        'window': stft_config['window_fn'],
        'center': stft_config['center']
    }


def main():
    # Paths
    audio_file = Path(__file__).parent / "mixture_4ch.wav"
    config_file = Path(__file__).parent.parent / "configs" / "constraint.yaml"
    
    print(f"Loading audio file: {audio_file}")
    print(f"Loading config file: {config_file}")
    
    # Load configuration
    config = load_config(config_file)
    
    # Load audio file
    audio_data, fs = load_wav_best_effort(str(audio_file))
    print(f"Audio shape: {audio_data.shape}, Sample rate: {fs} Hz")
    
    # Get STFT parameters from config
    stft_params = get_stft_params_from_yaml(config)
    print(f"STFT parameters: {stft_params}")
    
    # Compute STFT
    print("Computing STFT...")
    X, freqs, times = stft_multi(audio_data, fs, **stft_params)
    print(f"STFT shape: {X.shape}")
    print(f"Frequency bins: {len(freqs)}, Time frames: {len(times)}")
    
    feats = compute_mag_phase_cos_sin(X)
    print(f"ILD and IPD shape: {feats[:,:,:].shape}")

    det = VoiceActivityDetector(threshold=0.2, frame_rate=fs)
    probs = det.vad_probs_same_frame_count(
    audio_data[:, 0], fs,
    hop_ms=10.0,
    base_win_ms=32.0,   # this defines the centers you already use for the STFT
    big_win_ms=200,   # make the window bigger
    n_frames=times.shape[0],  # force same count (650)
    pad_mode="constant"  # or "reflect"
)

    print(f"VAD probabilities shape: {probs.shape}")
    print(f"VAD probabilities range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"Mean VAD probability: {probs.mean():.3f}")
    print(f"VAD probabilities: {probs}")

    # probs[k] corresponds to your STFT's frame k.


    # Get microphone configuration
    mic_xy = get_mic_config_from_yaml(config)
    print(f"Microphone positions (m): {mic_xy}")
    
    # Get SRP parameters from config
    az_res_deg = 1.0  # Default resolution
    c = 343.0  # Speed of sound
    
    # Compute regular SRP-PHAT map
    print("Computing regular SRP-PHAT map...")
    result_regular = estimate_azimuths_360_from_stft(
        X, freqs, mic_xy,
        method="srp",
        az_res_deg=az_res_deg,
        peak_kwargs={
            'top_k': 5,  # Get top 5 peaks first
            'min_separation_deg': 10.0,
            'min_prominence': 0.05
        }
    )
    
    # Compute VAD-weighted SRP-PHAT map using actual VAD probabilities
    print("Computing VAD-weighted SRP-PHAT map...")
    az_grid_vad, srp_map_vad = srp_weighted_VAD_phat_azimuth_map_360_from_stft(
        X, freqs, mic_xy, probs,
        az_res_deg=az_res_deg
    )
    
    # Find peaks in VAD-weighted map
    from mirokai_doa.srp import pick_peaks_360_robust
    peaks_vad = pick_peaks_360_robust(
        az_grid_vad, srp_map_vad,
        top_k=5,
        min_separation_deg=10.0,
        min_prominence=0.05
    )
    
    # Extract results from both methods
    azimuths_regular = result_regular['azimuths']
    az_grid = result_regular['az_grid_deg']
    srp_map = result_regular['map']
    
    # VAD-weighted results are already extracted above
    
    print(f"\n=== REGULAR SRP-PHAT ===")
    print(f"Found {len(azimuths_regular)} peaks (top 5):")
    for i, peak in enumerate(azimuths_regular):
        print(f"  Peak {i+1}: {peak['azimuth_deg']:.1f}° (score: {peak['score']:.3f}, prominence: {peak['prominence']:.3f})")
    
    print(f"\n=== VAD-WEIGHTED SRP-PHAT ===")
    print(f"Found {len(peaks_vad)} peaks (top 5):")
    for i, peak in enumerate(peaks_vad):
        print(f"  Peak {i+1}: {peak['azimuth_deg']:.1f}° (score: {peak['score']:.3f}, prominence: {peak['prominence']:.3f})")
    
    # Select 2 peaks with highest confidence for both methods
    if len(azimuths_regular) >= 2:
        sorted_peaks_regular = sorted(azimuths_regular, key=lambda p: p['score'], reverse=True)
        selected_peaks_regular = sorted_peaks_regular[:2]
    else:
        selected_peaks_regular = azimuths_regular
        
    if len(peaks_vad) >= 2:
        sorted_peaks_vad = sorted(peaks_vad, key=lambda p: p['score'], reverse=True)
        selected_peaks_vad = sorted_peaks_vad[:2]
    else:
        selected_peaks_vad = peaks_vad
    
    print(f"\n=== SELECTED PEAKS (Regular) ===")
    for i, peak in enumerate(selected_peaks_regular):
        print(f"  Selected {i+1}: {peak['azimuth_deg']:.1f}° (score: {peak['score']:.3f})")
    
    print(f"\n=== SELECTED PEAKS (VAD-Weighted) ===")
    for i, peak in enumerate(selected_peaks_vad):
        print(f"  Selected {i+1}: {peak['azimuth_deg']:.1f}° (score: {peak['score']:.3f})")
    
    # Real values for comparison
    true_azimuths = [90.0, 150.0]
    
    # Plot both SRP-PHAT maps for comparison
    print("Creating plot...")
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Regular SRP-PHAT map
    plt.subplot(2, 2, 1)
    plt.plot(az_grid, srp_map, 'b-', linewidth=1.5, label='Regular SRP-PHAT Map')
    
    # Mark all detected peaks in red
    for i, peak in enumerate(azimuths_regular):
        plt.axvline(peak['azimuth_deg'], color='red', linestyle='--', alpha=0.5, 
                   label=f"Peak {i+1}: {peak['azimuth_deg']:.1f}°" if i == 0 else f"Peak {i+1}: {peak['azimuth_deg']:.1f}°")
    
    # Highlight the 2 selected peaks in green
    for i, peak in enumerate(selected_peaks_regular):
        plt.axvline(peak['azimuth_deg'], color='green', linestyle='-', linewidth=2.0, alpha=0.9, 
                   label=f"Selected {i+1}: {peak['azimuth_deg']:.1f}°" if i == 0 else f"Selected {i+1}: {peak['azimuth_deg']:.1f}°")
    
    # Mark the real values in blue
    for i, true_az in enumerate(true_azimuths):
        plt.axvline(true_az, color='blue', linestyle=':', linewidth=3.0, alpha=0.8, 
                   label=f"True {i+1}: {true_az:.1f}°" if i == 0 else f"True {i+1}: {true_az:.1f}°")
    
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('SRP-PHAT Score')
    plt.title('Regular SRP-PHAT Map')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 360)
    
    # Plot 2: VAD-weighted SRP-PHAT map
    plt.subplot(2, 2, 2)
    plt.plot(az_grid_vad, srp_map_vad, 'b-', linewidth=1.5, label='VAD-Weighted SRP-PHAT Map')
    
    # Mark all detected peaks in red
    for i, peak in enumerate(peaks_vad):
        plt.axvline(peak['azimuth_deg'], color='red', linestyle='--', alpha=0.5, 
                   label=f"Peak {i+1}: {peak['azimuth_deg']:.1f}°" if i == 0 else f"Peak {i+1}: {peak['azimuth_deg']:.1f}°")
    
    # Highlight the 2 selected peaks in green
    for i, peak in enumerate(selected_peaks_vad):
        plt.axvline(peak['azimuth_deg'], color='green', linestyle='-', linewidth=2.0, alpha=0.9, 
                   label=f"Selected {i+1}: {peak['azimuth_deg']:.1f}°" if i == 0 else f"Selected {i+1}: {peak['azimuth_deg']:.1f}°")
    
    # Mark the real values in blue
    for i, true_az in enumerate(true_azimuths):
        plt.axvline(true_az, color='blue', linestyle=':', linewidth=3.0, alpha=0.8, 
                   label=f"True {i+1}: {true_az:.1f}°" if i == 0 else f"True {i+1}: {true_az:.1f}°")
    
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('SRP-PHAT Score')
    plt.title('VAD-Weighted SRP-PHAT Map')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 360)
    
    # Plot 3: Regular SRP-PHAT polar plot
    plt.subplot(2, 2, 3, projection='polar')
    plt.plot(np.deg2rad(az_grid), srp_map, 'b-', linewidth=1.5, label='Regular SRP-PHAT Map')
    
    # Mark all peaks on polar plot in red
    for i, peak in enumerate(azimuths_regular):
        plt.plot(np.deg2rad(peak['azimuth_deg']), peak['score'], 'ro', markersize=6, alpha=0.7,
                label=f"Peak {i+1}: {peak['azimuth_deg']:.1f}°" if i == 0 else f"Peak {i+1}: {peak['azimuth_deg']:.1f}°")
    
    # Highlight the 2 selected peaks on polar plot in green
    for i, peak in enumerate(selected_peaks_regular):
        plt.plot(np.deg2rad(peak['azimuth_deg']), peak['score'], 'go', markersize=10, 
                label=f"Selected {i+1}: {peak['azimuth_deg']:.1f}°" if i == 0 else f"Selected {i+1}: {peak['azimuth_deg']:.1f}°")
    
    # Mark the real values on polar plot in blue
    for i, true_az in enumerate(true_azimuths):
        true_score = np.interp(true_az, az_grid, srp_map)
        plt.plot(np.deg2rad(true_az), true_score, 'bo', markersize=12, 
                label=f"True {i+1}: {true_az:.1f}°" if i == 0 else f"True {i+1}: {true_az:.1f}°")

    plt.title('Regular SRP-PHAT (Polar)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.grid(True)
    
    # Plot 4: VAD-weighted SRP-PHAT polar plot
    plt.subplot(2, 2, 4, projection='polar')
    plt.plot(np.deg2rad(az_grid_vad), srp_map_vad, 'b-', linewidth=1.5, label='VAD-Weighted SRP-PHAT Map')
    
    # Mark all peaks on polar plot in red
    for i, peak in enumerate(peaks_vad):
        plt.plot(np.deg2rad(peak['azimuth_deg']), peak['score'], 'ro', markersize=6, alpha=0.7,
                label=f"Peak {i+1}: {peak['azimuth_deg']:.1f}°" if i == 0 else f"Peak {i+1}: {peak['azimuth_deg']:.1f}°")
    
    # Highlight the 2 selected peaks on polar plot in green
    for i, peak in enumerate(selected_peaks_vad):
        plt.plot(np.deg2rad(peak['azimuth_deg']), peak['score'], 'go', markersize=10, 
                label=f"Selected {i+1}: {peak['azimuth_deg']:.1f}°" if i == 0 else f"Selected {i+1}: {peak['azimuth_deg']:.1f}°")
    
    # Mark the real values on polar plot in blue
    for i, true_az in enumerate(true_azimuths):
        true_score = np.interp(true_az, az_grid_vad, srp_map_vad)
        plt.plot(np.deg2rad(true_az), true_score, 'bo', markersize=12, 
                label=f"True {i+1}: {true_az:.1f}°" if i == 0 else f"True {i+1}: {true_az:.1f}°")

    plt.title('VAD-Weighted SRP-PHAT (Polar)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = Path(__file__).parent / "srp_phat_map.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Also show the plot
    plt.show()
    
    print("Script completed successfully!")





if __name__ == "__main__":
    main()
