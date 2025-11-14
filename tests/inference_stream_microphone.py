#!/usr/bin/env python3
"""
Real-time streaming inference from microphone array with interactive DOA visualization.
Uses PyAudio to capture audio from Respeaker/Seeed mic array and processes in real-time.
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
from typing import Optional, Dict, List
import time
import argparse
import pyaudio
import queue
import threading

# Check CUDA availability (silent)

from mirokai_doa.vad import VoiceActivityDetector


# Audio conversion utilities
def byte_to_float(data: bytes) -> np.ndarray:
    """Convert 16-bit PCM bytes to float32 array in range [-1, 1]."""
    samples = np.frombuffer(data, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def float2pcm(audio: np.ndarray) -> np.ndarray:
    """Convert float32 array in range [-1, 1] to 16-bit PCM int16."""
    # Clip to valid range
    audio = np.clip(audio, -1.0, 1.0)
    # Convert to int16
    return (audio * 32767.0).astype(np.int16)

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
    Interactive streaming DOA visualization for real-time microphone input.
    Updates plot in real-time as audio is processed.
    """
    
    def __init__(self, max_history: int = 200, update_interval_ms: int = 100):
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
        self._last_event_type = 'unknown'  # Store last event type
        self._last_histogram = None  # Store last histogram
        self._last_histogram_bins = None  # Store last histogram bins
        
        # Initialize figure
        self.fig = plt.figure(figsize=(18, 12))
        self.gs = self.fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(self.gs[0, :])  # Azimuth over time
        self.ax2 = self.fig.add_subplot(self.gs[1, 0])  # Confidence
        self.ax3 = self.fig.add_subplot(self.gs[1, 1])  # VAD
        self.ax4 = self.fig.add_subplot(self.gs[2, :], projection='polar')  # Polar plot
        self.ax5 = self.fig.add_subplot(self.gs[3, :])  # Histogram
        
        # Initialize plots
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=2, label='Predicted Azimuth')
        self.line2, = self.ax2.plot([], [], 'g-', linewidth=2, label='Confidence')
        self.line3, = self.ax3.plot([], [], 'r-', linewidth=2, label='VAD')
        
        # Polar plot for current sources
        self.polar_scatter = None
        
        # Histogram plot
        self.histogram_bars = None
        self.histogram_bins = None
        
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
        self.ax1.set_title('DOA Prediction Over Time (Real-time Microphone)', fontsize=12, fontweight='bold')
        self.ax1.set_ylim(0, 360)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='upper right')
        
        # Confidence
        self.ax2.set_xlabel('Time (seconds)', fontsize=11)
        self.ax2.set_ylabel('Confidence', fontsize=11)
        self.ax2.set_title('Confidence Over Time', fontsize=12, fontweight='bold')
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper right')
        
        # VAD
        self.ax3.set_xlabel('Time (seconds)', fontsize=11)
        self.ax3.set_ylabel('VAD Probability', fontsize=11)
        self.ax3.set_title('Voice Activity Detection (Threshold: 0.12)', fontsize=12, fontweight='bold')
        self.ax3.set_ylim(0, 1)
        self.ax3.grid(True, alpha=0.3)
        # Add threshold line for voice activity
        self.ax3.axhline(y=0.12, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Voice Activity Threshold (0.12)')
        self.ax3.legend(loc='upper right')
        
        # Polar plot
        self.ax4.set_title('Current DOA Sources (Polar View)', fontsize=12, fontweight='bold', pad=20)
        self.ax4.set_theta_zero_location('N')  # 0° at top
        self.ax4.set_theta_direction(-1)  # Clockwise
        self.ax4.set_thetalim(0, 2 * np.pi)
        self.ax4.set_ylim(0, 1)
        self.ax4.set_yticklabels([])  # Remove radial labels
        # Draw circle at radius 1 (static background)
        circle = Circle((0, 0), 1, fill=False, color='gray', linestyle='--', linewidth=1)
        self.ax4.add_patch(circle)
        
        # Histogram plot
        self.ax5.set_xlabel('Azimuth (degrees)', fontsize=11)
        self.ax5.set_ylabel('Probability', fontsize=11)
        self.ax5.set_title('DOA Histogram (Real-time)', fontsize=12, fontweight='bold')
        self.ax5.set_xlim(0, 360)
        self.ax5.set_ylim(0, 1)
        self.ax5.grid(True, alpha=0.3)
        
    def update(self, results: Dict, current_time: float):
        """
        Update visualization with new inference results.
        
        Args:
            results: Results dictionary from InferenceModel.inference()
            current_time: Current time in seconds
        """
        azimuth = results.get('azimuth_deg')
        confidence = results.get('confidence', 0.0)
        vad = results.get('vad_probability', results.get('vad_mean', 0.0))
        event_type = results.get('event_type', 'unknown')
        sources = results.get('sources', [])
        histogram = results.get('histogram')
        histogram_bins = results.get('histogram_bins')
        
        # Store event type and histogram for display
        self._last_event_type = event_type
        self._last_histogram = histogram
        self._last_histogram_bins = histogram_bins
        
        # Append to history
        self.time_history.append(current_time)
        self.azimuth_history.append(azimuth if azimuth is not None else np.nan)
        self.confidence_history.append(confidence)
        self.vad_history.append(vad)
        self.sources_history.append(sources)
        
        # Keep only recent history
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
        
        # Convert to numpy arrays for easier manipulation
        times = np.array(self.time_history)
        azimuths = np.array(self.azimuth_history)
        confidences = np.array(self.confidence_history)
        vads = np.array(self.vad_history)
        
        # Update azimuth plot
        self.line1.set_data(times, azimuths)
        if len(times) > 1:
            self.ax1.set_xlim(max(0, times[-1] - 10), times[-1] + 1)  # Show last 10 seconds
        
        # Update confidence plot
        self.line2.set_data(times, confidences)
        if len(times) > 1:
            self.ax2.set_xlim(max(0, times[-1] - 10), times[-1] + 1)
        
        # Update VAD plot
        self.line3.set_data(times, vads)
        if len(times) > 1:
            self.ax3.set_xlim(max(0, times[-1] - 10), times[-1] + 1)
        
        # Update text with current prediction and event type
        if len(self.sources_history) > 0:
            current_sources = self.sources_history[-1]
            event_type = self._last_event_type
            vad_prob = vads[-1] if len(vads) > 0 else 0.0
            
            if current_sources:
                source_text = f"Event: {event_type.upper()}\n"
                source_text += f"VAD: {vad_prob:.3f}\n"
                source_text += "Sources:\n"
                for i, src in enumerate(current_sources):
                    source_text += f"  Source {i+1}: {src['azimuth_deg']:.1f}° (conf={src['score']:.3f})\n"
            else:
                azimuth = azimuths[-1] if not np.isnan(azimuths[-1]) else None
                source_text = f"Event: {event_type.upper()}\n"
                source_text += f"VAD: {vad_prob:.3f}\n"
                if azimuth is not None:
                    source_text += f"Azimuth: {azimuth:.1f}°\n"
                    source_text += f"Conf: {confidences[-1]:.3f}"
                else:
                    source_text += "No azimuth detected"
            self.text_azimuth.set_text(source_text)
        
        # Update polar plot with current sources (only clear if needed)
        if not hasattr(self, '_polar_artists'):
            self._polar_artists = []
        else:
            # Remove old artists
            for artist in self._polar_artists:
                artist.remove()
            self._polar_artists = []
        
        if len(self.sources_history) > 0:
            current_sources = self.sources_history[-1]
            colors = ['red', 'green', 'orange', 'purple', 'brown']
            for i, source in enumerate(current_sources):
                azimuth_rad = np.deg2rad(source['azimuth_deg'])
                score = source['score']
                color = colors[i % len(colors)]
                # Plot as arrow/line from center
                line, = self.ax4.plot([0, azimuth_rad], [0, score], 'o-', color=color, 
                             linewidth=3, markersize=8, label=f"Source {i+1}: {source['azimuth_deg']:.1f}°")
                self._polar_artists.append(line)
                # Add text label
                text = self.ax4.text(azimuth_rad, score + 0.1, f"{source['azimuth_deg']:.0f}°", 
                             ha='center', va='bottom', fontsize=10, color=color, fontweight='bold')
                self._polar_artists.append(text)
        
        # Update histogram plot (only clear bars, not entire axis)
        if not hasattr(self, '_hist_bars'):
            self._hist_bars = None
            self._hist_vlines = []
            self._hist_text = None
        
        # Remove old histogram elements
        if self._hist_bars is not None:
            self._hist_bars.remove()
            self._hist_bars = None
        for vline in self._hist_vlines:
            vline.remove()
        self._hist_vlines = []
        if self._hist_text is not None:
            self._hist_text.remove()
            self._hist_text = None
        
        if self._last_histogram is not None and self._last_histogram_bins is not None:
            histogram = self._last_histogram
            bins = self._last_histogram_bins
            
            if histogram.sum() > 0:
                # Plot histogram as bar chart
                bin_width = 360.0 / len(histogram)
                self._hist_bars = self.ax5.bar(bins, histogram, width=bin_width, 
                             edgecolor='black', linewidth=0.5, alpha=0.7, color='steelblue',
                             label='Histogram')
                
                # Mark detected sources
                if len(self.sources_history) > 0:
                    current_sources = self.sources_history[-1]
                    colors = ['red', 'green', 'orange', 'purple', 'brown']
                    for i, source in enumerate(current_sources):
                        color = colors[i % len(colors)]
                        vline = self.ax5.axvline(x=source['azimuth_deg'], color=color, linestyle='--', 
                                        linewidth=2, alpha=0.7, 
                                        label=f"Source {i+1}: {source['azimuth_deg']:.1f}°")
                        self._hist_vlines.append(vline)
                
                # Mark primary azimuth if available
                if len(self.azimuth_history) > 0 and not np.isnan(self.azimuth_history[-1]):
                    primary_az = self.azimuth_history[-1]
                    vline = self.ax5.axvline(x=primary_az, color='red', linestyle='-', 
                                    linewidth=2, alpha=0.5, label=f'Primary: {primary_az:.1f}°')
                    self._hist_vlines.append(vline)
                
                self.ax5.set_ylim(0, max(histogram.max() * 1.1, 0.1))
            else:
                self._hist_text = self.ax5.text(180, 0.5, 'No histogram data', 
                             ha='center', va='center', fontsize=12, 
                             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
                self.ax5.set_ylim(0, 1)
        else:
            self._hist_text = self.ax5.text(180, 0.5, 'Waiting for histogram data...', 
                         ha='center', va='center', fontsize=12,
                         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            self.ax5.set_ylim(0, 1)
        
        # Force redraw - use draw() for immediate update, but it's still fast now
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def find_respeaker_device(p: pyaudio.PyAudio) -> Optional[int]:
    """Find Respeaker device index."""
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            name = info['name'].lower()
            if 'respeaker' in name or 'seeed' in name or '2886' in name:
                return i
    return None


def list_audio_devices():
    """List available audio input devices."""
    p = pyaudio.PyAudio()
    print("\nAvailable audio input devices:")
    print("-" * 80)
    respeaker_idx = find_respeaker_device(p)
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            marker = " <-- Respeaker (auto-detected)" if i == respeaker_idx else ""
            print(f"Device {i}: {info['name']}{marker}")
            print(f"  Channels: {info['maxInputChannels']}, "
                  f"Sample Rate: {info['defaultSampleRate']:.0f} Hz")
            print()
    p.terminate()


def _chunk_to_floatarray(data: bytes, channels: int) -> np.ndarray:
    """
    Convert an interleaved audio byte-chunk to a (channels, samples) float array.
    
    Args:
        data: Interleaved audio bytes (16-bit PCM)
        channels: Number of channels
    
    Returns:
        np.ndarray: Shape (channels, n_samples) with float32 values in range [-1, 1]
    """
    # Convert bytes to float
    float_data = byte_to_float(data)  # shape (n_samples * C,)
    # Reshape to (samples, channels) then transpose to (channels, samples)
    return float_data.reshape(-1, channels).T  # shape (C, n_samples)


def stream_inference_from_microphone(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device_index: Optional[int] = None,
    sample_rate: int = 16000,
    window_ms: int = 200,
    hop_ms: int = 100,
    max_sources: int = 3,
    chunk_size: int = 5600
):
    """
    Stream audio from microphone and run real-time DOA inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file (optional, defaults to configs/train.yaml)
        device_index: PyAudio device index (None to auto-detect Respeaker)
        sample_rate: Audio sample rate (Hz)
        window_ms: Analysis window size in milliseconds
        hop_ms: Hop size in milliseconds
        max_sources: Maximum number of sources to detect
        chunk_size: PyAudio chunk size (frames per buffer)
        
    """
    # Load model
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "train.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Force CUDA if available for real-time performance
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = InferenceModel(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device  # Explicitly set device
    )
    
    # Initialize visualizer
    visualizer = StreamingDOAVisualizer(max_history=200)
    plt.ion()  # Turn on interactive mode
    plt.show(block=False)
    
    # Calculate frame sizes
    window_samples = int(sample_rate * window_ms / 1000)
    hop_samples = int(sample_rate * hop_ms / 1000)
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Auto-detect Respeaker if device_index not specified
    if device_index is None:
        device_index = find_respeaker_device(p)
        if device_index is None:
            list_audio_devices()
            p.terminate()
            return
    
    device_info = p.get_device_info_by_index(device_index)
    device_channels = device_info['maxInputChannels']
    
    # Constants matching your implementation
    FORMAT = pyaudio.paInt16
    CHANNELS = 6  # Respeaker has 6 channels
    DSP_CHANNEL = 0
    ECHO_CHANNEL = 5
    RAW_CHANNELS = slice(1, 5)  # Channels 1, 2, 3, 4 (indices 1, 2, 3, 4)
    
    # Audio buffer for accumulating samples
    audio_buffer = np.zeros((4, window_samples), dtype=np.float32)
    buffer_fill = 0
    start_time = time.time()
    
    # Queue for audio chunks (callback-based streaming)
    audio_queue = queue.Queue()
    stream_closed = False
    
    def _fill_buffer(in_data, frame_count, time_info, status_flags):
        """Callback function to continuously collect data from the audio stream."""
        if not stream_closed:
            audio_queue.put(in_data)
        return None, pyaudio.paContinue
    
    # Open audio stream with callback
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk_size,
        stream_callback=_fill_buffer
    )
    
    stream.start_stream()
    
    
    try:
        while True:
            # Get chunk from queue (blocking)
            try:
                data = audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # Convert bytes to float array (channels, samples)
            chunk_all_channels = _chunk_to_floatarray(data, CHANNELS)  # (6, chunk_size)
            
            # Extract raw microphone channels (1-4, which is slice(1, 5) = indices 1, 2, 3, 4)
            audio_chunk = chunk_all_channels[RAW_CHANNELS, :]  # (4, chunk_size)
            
            chunk_samples = audio_chunk.shape[1]
            current_time = time.time() - start_time
            
            # Add chunk to buffer
            if buffer_fill + chunk_samples <= window_samples:
                # Add to buffer
                audio_buffer[:, buffer_fill:buffer_fill + chunk_samples] = audio_chunk
                buffer_fill += chunk_samples
            else:
                # Fill remaining space and process
                remaining = window_samples - buffer_fill
                if remaining > 0:
                    audio_buffer[:, buffer_fill:] = audio_chunk[:, :remaining]
                    buffer_fill = window_samples
                
                # Process when buffer is full
                if buffer_fill >= window_samples:
                    t_inf_start = time.perf_counter()
                    
                    # Run inference on full window
                    results = model.inference(
                        audio_buffer,
                        return_histogram=True,
                        return_per_frame=False,
                        max_sources=max_sources,
                        detect_multi_source=True
                    )
                    t_inf = time.perf_counter() - t_inf_start
                    
                    t_viz_start = time.perf_counter()
                    # Update visualization
                    visualizer.update(results, current_time)
                    t_viz = time.perf_counter() - t_viz_start
                    
                    # Get timing information
                    timing = results.get('timing', {})
                    
                    # Print timing breakdown
                    print(f"[{current_time:.2f}s] "
                          f"TOTAL={t_inf*1000:.1f}ms | "
                          f"FEAT={timing.get('features', 0):.1f}ms | "
                          f"MODEL={timing.get('model', 0):.1f}ms | "
                          f"HIST={timing.get('histogram', 0):.1f}ms | "
                          f"DETECT={timing.get('detect', 0):.1f}ms | "
                          f"VIZ={t_viz*1000:.1f}ms")
                    
                    # Shift buffer by hop size for sliding window
                    audio_buffer[:, :-hop_samples] = audio_buffer[:, hop_samples:]
                    buffer_fill = window_samples - hop_samples
                    
                    # Add remaining part of chunk to buffer if any
                    if chunk_samples > remaining:
                        remaining_from_chunk = min(chunk_samples - remaining, hop_samples)
                        if remaining_from_chunk > 0:
                            audio_buffer[:, buffer_fill:buffer_fill + remaining_from_chunk] = \
                                audio_chunk[:, remaining:remaining + remaining_from_chunk]
                            buffer_fill += remaining_from_chunk
            
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        stream_closed = True
        stream.stop_stream()
        stream.close()
        p.terminate()
        plt.close('all')


def main():
    parser = argparse.ArgumentParser(
        description="Real-time DOA inference from microphone array",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default microphone
  python -m tests.inference_stream_microphone --checkpoint models/basic/2025-11-06_22-37-00-6a5fbc92/last.pt
  
  # Use specific device
  python -m tests.inference_stream_microphone --checkpoint models/basic/2025-11-06_22-37-00-6a5fbc92/last.pt --device-index 2
  
  # List available devices
  python -m tests.inference_stream_microphone --list-devices
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=False,
        help='Path to model checkpoint file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: configs/train.yaml)'
    )
    parser.add_argument(
        '--device-index',
        type=int,
        default=None,
        help='PyAudio device index (None for default)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Audio sample rate in Hz (default: 16000)'
    )
    parser.add_argument(
        '--window-ms',
        type=int,
        default=200,
        help='Analysis window size in milliseconds (default: 200)'
    )
    parser.add_argument(
        '--hop-ms',
        type=int,
        default=100,
        help='Hop size in milliseconds (default: 100)'
    )
    parser.add_argument(
        '--max-sources',
        type=int,
        default=3,
        help='Maximum number of sources to detect (default: 3)'
    )
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available audio input devices and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        return
    
    if args.checkpoint is None:
        parser.error("--checkpoint is required (unless using --list-devices)")
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint file not found: {checkpoint_path}")
    
    stream_inference_from_microphone(
        checkpoint_path=str(checkpoint_path),
        config_path=args.config,
        device_index=args.device_index,
        sample_rate=args.sample_rate,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        max_sources=args.max_sources
    )


if __name__ == "__main__":
    main()

