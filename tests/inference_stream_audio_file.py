#!/usr/bin/env python3
"""
Streaming inference on audio file with interactive visualization.
Processes audio in 200ms windows with 100ms hop and updates visualization in real-time.
"""

import sys
from pathlib import Path

# Add src directory to Python path so mirokai_doa can be imported
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import yaml
import soundfile as sf
from typing import Optional, Dict, List
import time

from mirokai_doa.vad import VoiceActivityDetector

# Import InferenceModel - handle both direct execution and module execution
try:
    from tests.test_inference import InferenceModel
except ImportError:
    # If running as module, try relative import
    try:
        from .test_inference import InferenceModel
    except ImportError:
        # If that fails, add tests to path and import
        tests_dir = Path(__file__).parent
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))
        from test_inference import InferenceModel


class StreamingDOAVisualizer:
    """
    Interactive streaming DOA visualization.
    Updates plot in real-time as audio is processed.
    """
    
    def __init__(self, max_history: int = 100, update_interval_ms: int = 100):
        """
        Initialize visualizer.
        
        Args:
            max_history: Maximum number of time steps to keep in history
            update_interval_ms: Update interval in milliseconds
        """
        self.max_history = max_history
        self.update_interval_ms = update_interval_ms
        
        # Data storage
        self.time_history = []
        self.azimuth_history = []
        self.confidence_history = []
        self.vad_history = []
        self.sources_history = []  # List of source lists per time step
        
        # Initialize figure
        self.fig = plt.figure(figsize=(16, 10))
        self.gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(self.gs[0, :])  # Azimuth over time
        self.ax2 = self.fig.add_subplot(self.gs[1, 0])  # Confidence
        self.ax3 = self.fig.add_subplot(self.gs[1, 1])  # VAD
        self.ax4 = self.fig.add_subplot(self.gs[2, :], projection='polar')  # Polar plot
        
        # Initialize plots
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2, label='Predicted Azimuth')
        self.line2, = self.ax2.plot([], [], 'g-', linewidth=2, label='Confidence')
        self.line3, = self.ax3.plot([], [], 'r-', linewidth=2, label='VAD')
        
        # Polar plot for current sources
        self.polar_scatter = None
        
        # Setup axes
        self.setup_axes()
        
        # Text for current predictions
        self.text_azimuth = self.ax1.text(0.02, 0.98, '', transform=self.ax1.transAxes,
                                          fontsize=12, verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    def setup_axes(self):
        """Setup and format all axes."""
        # Azimuth over time
        self.ax1.set_xlabel('Time (seconds)', fontsize=11)
        self.ax1.set_ylabel('Azimuth (degrees)', fontsize=11)
        self.ax1.set_title('DOA Prediction Over Time (Streaming)', fontsize=12, fontweight='bold')
        self.ax1.set_ylim(0, 360)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        # Confidence
        self.ax2.set_xlabel('Time (seconds)', fontsize=11)
        self.ax2.set_ylabel('Confidence', fontsize=11)
        self.ax2.set_title('Prediction Confidence', fontsize=12, fontweight='bold')
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        
        # VAD
        self.ax3.set_xlabel('Time (seconds)', fontsize=11)
        self.ax3.set_ylabel('VAD Probability', fontsize=11)
        self.ax3.set_title('Voice Activity Detection', fontsize=12, fontweight='bold')
        self.ax3.set_ylim(0, 1)
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend()
        
        # Polar plot
        self.ax4.set_title('Current DOA Sources (Polar View)', fontsize=12, fontweight='bold', pad=20)
        self.ax4.set_theta_zero_location('N')  # 0° at top
        self.ax4.set_theta_direction(-1)  # Clockwise
        self.ax4.set_ylim(0, 1)
        
    def update(self, time_sec: float, azimuth: Optional[float], confidence: float, 
               vad_prob: float, sources: List[Dict]):
        """
        Update visualization with new data point.
        
        Args:
            time_sec: Current time in seconds
            azimuth: Predicted azimuth in degrees (None if no prediction)
            confidence: Prediction confidence
            vad_prob: VAD probability
            sources: List of detected sources
        """
        # Add to history
        self.time_history.append(time_sec)
        self.azimuth_history.append(azimuth if azimuth is not None else np.nan)
        self.confidence_history.append(confidence)
        self.vad_history.append(vad_prob)
        self.sources_history.append(sources)
        
        # Limit history
        if len(self.time_history) > self.max_history:
            self.time_history.pop(0)
            self.azimuth_history.pop(0)
            self.confidence_history.pop(0)
            self.vad_history.pop(0)
            self.sources_history.pop(0)
        
        # Update plots
        self.update_plots()
        
    def update_plots(self):
        """Update all plot elements."""
        if len(self.time_history) == 0:
            return
        
        # Convert to numpy arrays
        times = np.array(self.time_history)
        azimuths = np.array(self.azimuth_history)
        confidences = np.array(self.confidence_history)
        vads = np.array(self.vad_history)
        
        # Update azimuth plot
        self.line1.set_data(times, azimuths)
        if len(times) > 0:
            self.ax1.set_xlim(max(0, times[-1] - 5), times[-1] + 0.5)  # Show last 5 seconds
        
        # Update confidence plot
        self.line2.set_data(times, confidences)
        if len(times) > 0:
            self.ax2.set_xlim(max(0, times[-1] - 5), times[-1] + 0.5)
        
        # Update VAD plot
        self.line3.set_data(times, vads)
        if len(times) > 0:
            self.ax3.set_xlim(max(0, times[-1] - 5), times[-1] + 0.5)
        
        # Update polar plot with current sources
        self.ax4.clear()
        self.ax4.set_title('Current DOA Sources (Polar View)', fontsize=12, fontweight='bold', pad=20)
        self.ax4.set_theta_zero_location('N')
        self.ax4.set_theta_direction(-1)
        self.ax4.set_ylim(0, 1)
        
        if len(self.sources_history) > 0:
            current_sources = self.sources_history[-1]
            if current_sources:
                colors = ['red', 'green', 'orange', 'purple', 'brown']
                for i, source in enumerate(current_sources):
                    az_rad = np.deg2rad(source['azimuth_deg'])
                    score = source['score']
                    color = colors[i % len(colors)]
                    self.ax4.scatter([az_rad], [score], s=200, c=color, alpha=0.7, 
                                   label=f"Source {i+1}: {source['azimuth_deg']:.1f}°")
                    # Draw line from center
                    self.ax4.plot([az_rad, az_rad], [0, score], color=color, linewidth=2, alpha=0.5)
                self.ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Update text
        if len(self.azimuth_history) > 0 and not np.isnan(self.azimuth_history[-1]):
            current_az = self.azimuth_history[-1]
            current_conf = self.confidence_history[-1]
            current_vad = self.vad_history[-1]
            num_sources = len(self.sources_history[-1]) if self.sources_history else 0
            
            text = f"Time: {times[-1]:.2f}s\n"
            text += f"Azimuth: {current_az:.1f}°\n"
            text += f"Confidence: {current_conf:.3f}\n"
            text += f"VAD: {current_vad:.3f}\n"
            text += f"Sources: {num_sources}"
            self.text_azimuth.set_text(text)
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def show(self):
        """Show the plot."""
        plt.show()


def stream_inference_on_file(
    model: InferenceModel,
    audio_path: str,
    window_ms: float = 200.0,
    hop_ms: float = 100.0,
    max_sources: int = 3,
    save_output: Optional[str] = None
):
    """
    Stream inference on audio file with interactive visualization.
    
    Args:
        model: InferenceModel instance
        audio_path: Path to audio file
        window_ms: Window size in milliseconds
        hop_ms: Hop size in milliseconds
        max_sources: Maximum number of sources to detect
        save_output: Optional path to save output video/animation
    """
    print(f"Loading audio file: {audio_path}")
    mixture, sr = sf.read(audio_path)
    
    # Ensure multichannel
    if mixture.ndim == 1:
        mixture = np.tile(mixture[:, None], (1, 4))  # Mono to 4ch
    if mixture.shape[1] != 4:
        raise ValueError(f"Expected 4 channels, got {mixture.shape[1]}")
    
    mixture = mixture.T  # (4, T_samples)
    
    # Convert to samples
    window_samples = int(window_ms * sr / 1000.0)
    hop_samples = int(hop_ms * sr / 1000.0)
    
    total_samples = mixture.shape[1]
    total_time = total_samples / sr
    
    print(f"Audio: {total_time:.2f}s, {sr}Hz, {total_samples} samples")
    print(f"Window: {window_ms}ms ({window_samples} samples)")
    print(f"Hop: {hop_ms}ms ({hop_samples} samples)")
    print(f"Total windows: {(total_samples - window_samples) // hop_samples + 1}")
    
    # Initialize visualizer
    visualizer = StreamingDOAVisualizer(max_history=200)
    
    # Process audio in chunks
    start_idx = 0
    window_count = 0
    
    print("\nStarting streaming inference...")
    print("Close the plot window to stop.\n")
    
    plt.ion()  # Turn on interactive mode
    visualizer.show()
    
    try:
        while start_idx + window_samples <= total_samples:
            # Extract window
            chunk = mixture[:, start_idx:start_idx + window_samples]  # (4, window_samples)
            
            # Current time (center of window)
            current_time = (start_idx + window_samples / 2) / sr
            
            # Run inference
            results = model.inference(
                chunk,
                return_histogram=True,
                return_per_frame=False,
                max_sources=max_sources,
                detect_multi_source=True
            )
            
            # Extract results
            azimuth = results.get('azimuth_deg')
            confidence = results.get('confidence', 0.0)
            sources = results.get('sources', [])
            
            # Get VAD (average over window)
            # We need to compute VAD for this chunk
            X, freqs, feats, vad_probs = model.compute_features(chunk)
            vad_avg = vad_probs.mean() if len(vad_probs) > 0 else 0.0
            
            # Update visualization
            visualizer.update(
                time_sec=current_time,
                azimuth=azimuth,
                confidence=confidence,
                vad_prob=vad_avg,
                sources=sources
            )
            
            # Print progress
            if window_count % 10 == 0:
                azimuth_str = f"{azimuth:.1f}°" if azimuth is not None else "None"
                source_str = ", ".join([f"{s['azimuth_deg']:.1f}°" for s in sources]) if sources else "None"
                print(f"Time: {current_time:.2f}s | Azimuth: {azimuth_str} | "
                      f"Conf: {confidence:.3f} | VAD: {vad_avg:.3f} | Sources: [{source_str}]")
            
            # Move to next window
            start_idx += hop_samples
            window_count += 1
            
            # Small delay for visualization (optional)
            time.sleep(0.01)  # 10ms delay for smoother visualization
        
        print(f"\nCompleted processing {window_count} windows")
        print("Press Enter to close...")
        input()
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        plt.ioff()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stream inference on audio file with interactive visualization')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to config file')
    parser.add_argument('--model-name', type=str, help='Model name (auto-detected if not provided)')
    parser.add_argument('--window-ms', type=float, default=200.0, help='Window size in milliseconds')
    parser.add_argument('--hop-ms', type=float, default=100.0, help='Hop size in milliseconds')
    parser.add_argument('--max-sources', type=int, default=3, help='Maximum number of sources to detect')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    if args.checkpoint:
        print(f"Loading model from: {args.checkpoint}")
        model = InferenceModel(
            checkpoint_path=args.checkpoint,
            config=config,
            model_name=args.model_name
        )
    else:
        raise ValueError("--checkpoint is required")
    
    # Run streaming inference
    stream_inference_on_file(
        model=model,
        audio_path=args.audio,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        max_sources=args.max_sources
    )


if __name__ == "__main__":
    # Example usage (uncomment and modify paths)
    if len(sys.argv) == 1:
        # Default example
        from pathlib import Path
        import yaml
        
        checkpoint_path = "models/basic/2025-11-06_22-37-00-6a5fbc92/last.pt"
        config_path = Path(__file__).parent.parent / "configs" / "train.yaml"
        audio_path = Path(__file__).parent / "mixture_4ch.wav"
        
        if not audio_path.exists():
            audio_path = Path(__file__).parent / "sample_mix.wav"
        
        if not audio_path.exists():
            print("Error: Audio file not found. Please provide --audio argument.")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model = InferenceModel(
            checkpoint_path=checkpoint_path,
            config=config
        )
        
        stream_inference_on_file(
            model=model,
            audio_path=str(audio_path),
            window_ms=200.0,
            hop_ms=100.0,
            max_sources=3
        )
    else:
        main()

