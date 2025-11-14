#!/usr/bin/env python3
"""
Inference module for DOA estimation.
Computes features from audio mixture, runs model inference, and generates histogram of predictions.
"""

import sys
from pathlib import Path

# Add src directory to Python path so mirokai_doa can be imported
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, Union, List
import yaml
import matplotlib.pyplot as plt

from mirokai_doa.vad import VoiceActivityDetector
from mirokai_doa.features import stft_multi, compute_mag_phase_cos_sin
from mirokai_doa.srp import srp_phat_azimuth_map_360_from_stft, pick_peaks_360_robust
from mirokai_doa.utils_model import decode_angle_from_logits
from mirokai_doa.train_utils import load_checkpoint, build_model
from mirokai_doa.doa_model import DoAEstimator
from scipy.signal import find_peaks


def detect_model_type_from_checkpoint(checkpoint: Dict) -> Optional[str]:
    """
    Detect model type from checkpoint keys.
    
    Returns:
        Model type string ('doa', 'basic', 'scat', 'film', 'retin') or None if unknown
    """
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    keys = list(state_dict.keys())
    
    # DoAEstimator has keys like "c1.net.0.weight", "rnn.weight_ih_l0", "ff1.weight"
    if any('c1.net' in k or 'rnn.weight_ih_l0' in k or 'ff1.weight' in k for k in keys):
        return 'doa'
    
    # TFPoolClassifierNoCond has keys like "block1.0.weight", "mlp.0.weight"
    if any('block1.0.weight' in k or 'mlp.0.weight' in k for k in keys):
        return 'basic'
    
    # SCATTiny has keys like "backbone", "srp_proto", "cross"
    if any('srp_proto' in k or 'backbone' in k and 'cross' in str(keys) for k in keys):
        return 'scat'
    
    # FiLMMixerSRP has keys like "film", "blocks"
    if any('film' in k and 'blocks' in str(keys) for k in keys):
        return 'film'
    
    # ReTiNDoA has keys like "cell"
    if any('cell' in k for k in keys):
        return 'retin'
    
    return None


class InferenceModel:
    """
    Inference class for DOA estimation models.
    Handles feature computation, model inference, and result aggregation.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize inference model.
        
        Args:
            model: PyTorch model instance (or None if loading from checkpoint)
            checkpoint_path: Path to model checkpoint (.pth file)
            config: Configuration dictionary with feature/microphone settings
            device: Device to run inference on ('cpu' or 'cuda')
            model_name: Model type ('scat', 'film', 'retin', 'basic') if loading from checkpoint
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load configuration
        if config is None:
            # Try to load from default config
            config_path = Path(__file__).parent.parent / "configs" / "train.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                # Default config
                config = {
                    'features': {
                        'sr': 16000,
                        'win_s': 0.032,
                        'hop_s': 0.010,
                        'nfft': 1024,
                        'K': 72
                    },
                    'microphone': {
                        'positions': [
                            [0.0277, 0.0],
                            [0.0, 0.0277],
                            [-0.0277, 0.0],
                            [0.0, -0.0277]
                        ]
                    }
                }
        
        self.config = config
        self.features_cfg = config.get('features', {})
        self.mic_xy = np.array(config.get('microphone', {}).get('positions', []), dtype=np.float32)
        self.K = self.features_cfg.get('K', 72)
        
        # Initialize VAD detector
        # Note: threshold is used for binary decisions, but we use probabilities directly
        # So threshold doesn't matter much here, but we keep it for compatibility
        self.vad_detector = VoiceActivityDetector(
            threshold=self.features_cfg.get('vad_threshold', 0.5),
            frame_rate=self.features_cfg.get('sr', 16000)
        )
        self.vad_threshold = self.features_cfg.get('vad_threshold', 0.5)
        # Voice activity threshold (separate from VAD threshold for filtering)
        self.voice_activity_threshold = 0.12
        
        # Load or use provided model
        if checkpoint_path is not None:
            checkpoint = load_checkpoint(checkpoint_path)
            
            # Detect model type from checkpoint if not provided
            if model_name is None:
                detected_type = detect_model_type_from_checkpoint(checkpoint)
                if detected_type:
                    model_name = detected_type
                else:
                    raise ValueError("model_name must be provided when auto-detection fails")
            
            # Build model if not provided
            if model is None:
                # Handle DoAEstimator separately since it's not in build_model
                if model_name == 'doa':
                    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
                    # Try to infer num_mics and num_classes from checkpoint
                    # Check for num_mics (usually 12 for 4ch * 3 features)
                    num_mics = 12  # Default
                    num_classes = self.K  # Use K from config
                    # Check if we can infer from keys
                    if 'c1.net.0.weight' in state_dict:
                        num_mics = state_dict['c1.net.0.weight'].shape[1]
                    if 'ff2.weight' in state_dict:
                        num_classes = state_dict['ff2.weight'].shape[0]
                    model = DoAEstimator(num_mics=num_mics, num_classes=num_classes)
                else:
                    model = build_model(model_name, config)
            
            # Load weights with error handling
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                # Try with strict=False for partial loading
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys or unexpected_keys:
                    raise RuntimeError(f"Failed to load checkpoint. Missing: {len(missing_keys)} keys, Unexpected: {len(unexpected_keys)} keys")
        
        if model is None:
            raise ValueError("Either model instance or checkpoint_path must be provided")
        
        self.model = model.to(self.device)
        self.model.eval()
        
        # Check if model needs SRP input
        # Models like SCATTiny, FiLMMixerSRP need SRP, others don't
        self.needs_srp = hasattr(self.model, 'forward') and self._check_model_signature()
    
    def _check_model_signature(self) -> bool:
        """Check if model forward() requires SRP input."""
        import inspect
        sig = inspect.signature(self.model.forward)
        params = list(sig.parameters.keys())
        # If forward has 'srp' parameter, it needs SRP
        return 'srp' in params
    
    def compute_features(self, mixture: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute STFT features and VAD from audio mixture.
        
        Args:
            mixture: Audio mixture array, shape (4, T_samples) or (T_samples, 4)
        
        Returns:
            X: Complex STFT, shape (T_frames, 4, F)
            freqs: Frequency bins, shape (F,)
            feats: Features (mag + cos + sin), shape (T_frames, 12, F)
            vad_probs: VAD probabilities, shape (T_frames,)
        """
        # Ensure correct shape: (4, T_samples)
        if mixture.ndim == 1:
            raise ValueError("Mixture must be multichannel (4 channels)")
        if mixture.shape[0] != 4 and mixture.shape[1] == 4:
            mixture = mixture.T  # Transpose if (T, 4) -> (4, T)
        
        if mixture.shape[0] != 4:
            raise ValueError(f"Expected 4 channels, got {mixture.shape[0]}")
        
        # Convert to float32
        x4 = mixture.astype(np.float32)
        
        # STFT parameters
        sr = self.features_cfg.get('sr', 16000)
        win_s = self.features_cfg.get('win_s', 0.032)
        hop_s = self.features_cfg.get('hop_s', 0.010)
        nfft = self.features_cfg.get('nfft', 1024)
        
        # Compute STFT: input is (T_samples, 4), output is (T_frames, 4, F)
        X, freqs, times = stft_multi(
            x4.T,  # Transpose to (T_samples, 4)
            fs=sr,
            win_s=win_s,
            hop_s=hop_s,
            nfft=nfft,
            window="hann",
            center=True,
            pad_mode="reflect"
        )
        
        # Compute features: magnitude + cos(phase) + sin(phase)
        feats = compute_mag_phase_cos_sin(X, dtype=np.float32)  # (T_frames, 12, F)
        
        # Compute VAD on entire audio buffer (not per frame)
        # Use audio_forward to process the entire buffer in chunks
        audio_mono = x4[0]  # Use first channel for VAD
        audio_tensor = torch.from_numpy(audio_mono.copy()).unsqueeze(0)  # (1, T_samples)
        
        # Process entire buffer through VAD model
        vad_output = self.vad_detector.model.audio_forward(audio_tensor, sr, num_samples=512)
        # vad_output is (1, T_vad_frames) - get mean probability for entire buffer
        vad_prob_entire_buffer = vad_output.mean().item()
        
        # Use the same VAD probability for all STFT frames
        vad_probs = np.full(X.shape[0], vad_prob_entire_buffer, dtype=np.float32)
        
        return X, freqs, feats, vad_probs
    
    def compute_srp(self, X: np.ndarray, freqs: np.ndarray, vad_probs: np.ndarray) -> np.ndarray:
        """
        Compute SRP-PHAT features for each time frame.
        
        Args:
            X: Complex STFT, shape (T_frames, 4, F)
            freqs: Frequency bins, shape (F,)
            vad_probs: VAD probabilities, shape (T_frames,)
        
        Returns:
            srp: SRP-PHAT maps, shape (T_frames, K)
        """
        T_frames = X.shape[0]
        K = self.K
        srp_maps = np.zeros((T_frames, K), dtype=np.float32)
        
        # Compute SRP for each frame
        for t in range(T_frames):
            # Get single frame STFT: (1, 4, F)
            X_frame = X[t:t+1]  # (1, 4, F)
            
            # Compute SRP-PHAT map for this frame
            az_deg, score = srp_phat_azimuth_map_360_from_stft(
                X_frame,  # (1, 4, F)
                freqs,
                self.mic_xy,
                az_res_deg=360.0 / K,
                c=343.0
            )
            
            # Store SRP map
            srp_maps[t] = score
        
        return srp_maps
    
    def detect_multiple_sources(
        self,
        histogram: np.ndarray,
        bin_centers: np.ndarray,
        max_sources: int = 3,
        min_separation_deg: float = 15.0,
        min_prominence: float = 0.1
    ) -> List[Dict]:
        """
        Detect multiple sources from histogram using peak picking.
        Ensures no duplicate detections by enforcing minimum separation.
        
        Args:
            histogram: Probability distribution over azimuth bins (K,)
            bin_centers: Azimuth bin centers in degrees (K,)
            max_sources: Maximum number of sources to detect
            min_separation_deg: Minimum angular separation between sources
            min_prominence: Minimum prominence for a peak to be considered
        
        Returns:
            List of dicts with 'azimuth_deg', 'score', 'prominence' for each source
        """
        # Use peak picking on histogram
        peaks = pick_peaks_360_robust(
            bin_centers,
            histogram,
            top_k=max_sources * 2,  # Get more candidates initially
            min_separation_deg=min_separation_deg,
            min_prominence=min_prominence,
            refine_subdegree=True
        )
        
        # Filter out duplicates and ensure minimum separation
        if len(peaks) == 0:
            return []
        
        # Sort by score (descending)
        peaks = sorted(peaks, key=lambda x: x['score'], reverse=True)
        
        # Filter to ensure minimum separation (circular distance)
        filtered_peaks = []
        for peak in peaks:
            az = peak['azimuth_deg']
            is_duplicate = False
            
            # Check against already selected peaks
            for selected in filtered_peaks:
                sel_az = selected['azimuth_deg']
                # Circular distance
                dist = min(abs(az - sel_az), 360 - abs(az - sel_az))
                if dist < min_separation_deg:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_peaks.append(peak)
                if len(filtered_peaks) >= max_sources:
                    break
        
        return filtered_peaks
    
    def inference(
        self,
        mixture: np.ndarray,
        return_histogram: bool = True,
        return_per_frame: bool = False,
        max_sources: int = 3,
        detect_multi_source: bool = True
    ) -> Union[Dict, Tuple[Dict, np.ndarray]]:
        """
        Run inference on audio mixture.
        
        Args:
            mixture: Audio mixture array, shape (4, T_samples)
            return_histogram: If True, compute and return histogram of predictions
            return_per_frame: If True, return per-frame predictions
            max_sources: Maximum number of sources to detect (for multi-source mode)
            detect_multi_source: If True, detect multiple sources from histogram
        
        Returns:
            Dictionary with:
                - 'azimuth_deg': Predicted azimuth in degrees (if single prediction)
                - 'azimuth_bin': Predicted bin index (0 to K-1)
                - 'confidence': Prediction confidence
                - 'sources': List of detected sources (if detect_multi_source=True)
                - 'per_frame_azimuth': Per-frame azimuths (if return_per_frame=True)
                - 'per_frame_bins': Per-frame bin indices (if return_per_frame=True)
                - 'per_frame_topk': Top-K sources per frame (if return_per_frame=True)
                - 'histogram': Histogram of bin predictions (if return_histogram=True)
        """
        import time
        t_total_start = time.perf_counter()
        t_feat_start = time.perf_counter()
        
        with torch.no_grad():
            # Compute features
            X, freqs, feats, vad_probs = self.compute_features(mixture)
            t_feat = time.perf_counter() - t_feat_start
            
            T_frames, C_feat, F = feats.shape
            assert C_feat == 12, f"Expected 12 feature channels, got {C_feat}"
            
            # Process in batches of 25 frames
            batch_size = 25
            all_logits = []
            all_probs = []
            
            t_model_start = time.perf_counter()
            for start_idx in range(0, T_frames, batch_size):
                end_idx = min(start_idx + batch_size, T_frames)
                batch_feats = feats[start_idx:end_idx]  # (batch_T, 12, F)
                batch_T = batch_feats.shape[0]
                
                # Pad last batch to 25 frames if needed
                if batch_T < batch_size:
                    padding = np.zeros((batch_size - batch_T, C_feat, F), dtype=batch_feats.dtype)
                    batch_feats = np.concatenate([batch_feats, padding], axis=0)  # (25, 12, F)
                    pad_mask = np.ones(batch_T, dtype=bool)  # Track which frames are real
                else:
                    pad_mask = np.ones(batch_T, dtype=bool)
                
                # Prepare input tensors for this batch
                # Features need to be reshaped: (T, 12, F) -> (1, 12, T, F) for model
                feats_tensor = torch.from_numpy(batch_feats).float()  # (25, 12, F)
                feats_tensor = feats_tensor.permute(1, 0, 2).unsqueeze(0)  # (1, 12, 25, F)
                feats_tensor = feats_tensor.to(self.device)
                
                # Compute SRP if needed for this batch
                if self.needs_srp:
                    t_srp_start = time.perf_counter()
                    batch_X = X[start_idx:end_idx]  # (batch_T, 4, F)
                    batch_vad = vad_probs[start_idx:end_idx]  # (batch_T,)
                    srp_maps = self.compute_srp(batch_X, freqs, batch_vad)  # (batch_T, K)
                    t_srp = time.perf_counter() - t_srp_start
                    # Pad SRP if needed
                    if batch_T < batch_size:
                        srp_padding = np.zeros((batch_size - batch_T, self.K), dtype=srp_maps.dtype)
                        srp_maps = np.concatenate([srp_maps, srp_padding], axis=0)  # (25, K)
                    srp_tensor = torch.from_numpy(srp_maps).float().unsqueeze(0)  # (1, 25, K)
                    srp_tensor = srp_tensor.to(self.device)
                    
                    # Run model with SRP
                    t_forward_start = time.perf_counter()
                    batch_logits = self.model(feats_tensor, srp_tensor)  # (1, 25, K)
                    t_forward = time.perf_counter() - t_forward_start
                else:
                    # Models that don't need SRP (like DoAEstimator expect different input format)
                    # Check if model expects (B, T, F, C) format
                    t_forward_start = time.perf_counter()
                    if isinstance(self.model, DoAEstimator):
                        # DoAEstimator format: (B, T, F, C)
                        feats_reshaped = feats_tensor.permute(0, 2, 3, 1)  # (1, 25, F, 12)
                        batch_logits = self.model(feats_reshaped)  # (1, 25, K)
                    elif hasattr(self.model, 'expected_feat_per_timestep'):
                        # Other models with expected_feat_per_timestep: (B, T, F, C)
                        feats_reshaped = feats_tensor.permute(0, 2, 3, 1)  # (1, 25, F, 12)
                        batch_logits = self.model(feats_reshaped)  # (1, 25, K)
                    else:
                        # TFPoolClassifierNoCond format: (B, 12, T, F)
                        batch_logits = self.model(feats_tensor, None)  # (1, K) or (1, 25, K)
                    t_forward = time.perf_counter() - t_forward_start
                
                # Handle different output shapes
                if batch_logits.dim() == 2:
                    # Single prediction: (1, K) -> add time dimension
                    batch_logits = batch_logits.unsqueeze(1)  # (1, 1, K)
                
                # Get logits for this batch
                batch_logits = batch_logits.squeeze(0)  # (25, K) or (1, K)
                if batch_logits.dim() == 1:
                    batch_logits = batch_logits.unsqueeze(0)  # (1, K)
                
                # Remove padding from last batch
                if batch_T < batch_size:
                    batch_logits = batch_logits[:batch_T]  # (batch_T, K)
                
                # Apply softmax to get probabilities
                batch_probs = torch.softmax(batch_logits, dim=-1)  # (batch_T, K)
                
                all_logits.append(batch_logits)
                all_probs.append(batch_probs)
            
            t_model = time.perf_counter() - t_model_start
            
            # Concatenate all batches
            t_post_start = time.perf_counter()
            logits = torch.cat(all_logits, dim=0)  # (T_frames, K)
            probs = torch.cat(all_probs, dim=0)  # (T_frames, K)
            
            # Decode angles from logits
            theta_hat, conf = decode_angle_from_logits(
                logits.unsqueeze(0),  # (1, T, K)
                K=self.K,
                method="argmax"
            )
            theta_hat = theta_hat.squeeze(0)  # (T,)
            conf = conf.squeeze(0)  # (T,)
            
            # Convert to degrees
            azimuth_deg = np.rad2deg(theta_hat.cpu().numpy())  # (T,)
            azimuth_bin = logits.argmax(dim=-1).cpu().numpy()  # (T,)
            probs_np = probs.cpu().numpy()  # (T, K) - probabilities for histogram
            
            # Determine event type based on VAD probability of entire buffer
            # VAD probability is the same for all frames (computed on entire buffer)
            vad_prob_buffer = vad_probs[0] if len(vad_probs) > 0 else 0.0
            
            # Apply VAD masking (only consider frames above VAD threshold)
            vad_mask = vad_probs >= self.vad_threshold
            
            voiced_azimuths = azimuth_deg[vad_mask]
            voiced_bins = azimuth_bin[vad_mask]
            
            # Aggregate results
            results = {}
            results['vad_probability'] = float(vad_prob_buffer)
            
            # Determine event type: voice activity if VAD > 0.12, else sound event
            if vad_prob_buffer > self.voice_activity_threshold:
                results['event_type'] = 'voice_activity'
            else:
                results['event_type'] = 'sound_event'
            
            if len(voiced_azimuths) > 0:
                # Overall prediction (circular mean of voiced frames)
                cos_mean = np.cos(np.deg2rad(voiced_azimuths)).mean()
                sin_mean = np.sin(np.deg2rad(voiced_azimuths)).mean()
                overall_azimuth = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0
                overall_confidence = conf[vad_mask].mean().item()
                
                results['azimuth_deg'] = float(overall_azimuth)
                results['azimuth_bin'] = int(np.argmax(np.bincount(voiced_bins, minlength=self.K)))
                results['confidence'] = float(overall_confidence)
            else:
                results['azimuth_deg'] = None
                results['azimuth_bin'] = None
                results['confidence'] = 0.0
            
            if return_per_frame:
                results['per_frame_azimuth'] = azimuth_deg
                results['per_frame_bins'] = azimuth_bin
                results['per_frame_confidence'] = conf.cpu().numpy()
                results['per_frame_probs'] = probs_np  # (T, K) - probabilities for each frame
                results['vad_probs'] = vad_probs
                results['frame_times'] = np.arange(len(azimuth_deg)) * self.features_cfg.get('hop_s', 0.010)  # Time in seconds
                
                # Extract top-K sources per frame
                topk_values, topk_indices = torch.topk(probs, k=min(max_sources, self.K), dim=-1)  # (T, topk)
                per_frame_topk_azimuth = np.rad2deg(
                    (topk_indices.cpu().numpy() + 0.5) * (2.0 * np.pi / self.K)
                ) % 360.0  # (T, topk)
                per_frame_topk_probs = topk_values.cpu().numpy()  # (T, topk)
                
                results['per_frame_topk_azimuth'] = per_frame_topk_azimuth
                results['per_frame_topk_probs'] = per_frame_topk_probs
            
            t_hist_start = time.perf_counter()
            if return_histogram:
                # Compute histogram using softmax probabilities (weighted by VAD)
                if len(voiced_bins) > 0:
                    # Get probabilities for voiced frames only
                    voiced_probs = probs_np[vad_mask]  # (T_voiced, K)
                    
                    # Weight by VAD probabilities (optional - already filtered by vad_mask)
                    vad_weights = vad_probs[vad_mask]  # (T_voiced,)
                    
                    # Sum probabilities across all voiced frames, weighted by VAD
                    histogram = (voiced_probs * vad_weights[:, np.newaxis]).sum(axis=0)  # (K,)
                    
                    # Normalize
                    if histogram.sum() > 0:
                        histogram = histogram / histogram.sum()
                    else:
                        histogram = np.zeros(self.K, dtype=np.float32)
                else:
                    # No voiced frames - compute histogram from all frames instead
                    histogram = probs_np.sum(axis=0)  # (K,) - sum across all frames
                    if histogram.sum() > 0:
                        histogram = histogram / histogram.sum()
                    else:
                        histogram = np.zeros(self.K, dtype=np.float32)
                
                # Store histogram (raw, no smoothing)
                results['histogram'] = histogram
                results['histogram_bins'] = np.arange(self.K) * (360.0 / self.K)  # Bin centers in degrees
            else:
                histogram = None
            t_hist = time.perf_counter() - t_hist_start
            
            # Only detect sources if histogram has non-zero values
            t_detect_start = time.perf_counter()
            if return_histogram and histogram is not None and histogram.sum() > 0:
                # Detect multiple sources from raw histogram
                if detect_multi_source:
                    sources = self.detect_multiple_sources(
                        histogram,
                        results['histogram_bins'],
                        max_sources=max_sources,
                        min_separation_deg=15.0,
                        min_prominence=0.05
                    )
                    results['sources'] = sources
                    results['num_sources'] = len(sources)
                    
                    # Update overall azimuth from histogram if we have sources
                    if len(sources) > 0:
                        # Use the highest scoring source as primary azimuth
                        primary_source = sources[0]
                        results['azimuth_deg'] = float(primary_source['azimuth_deg'])
                        results['azimuth_bin'] = int(np.round(primary_source['azimuth_deg'] / (360.0 / self.K))) % self.K
                        results['confidence'] = float(primary_source['score'])
                else:
                    # Single source: use peak of histogram
                    peak_bin = np.argmax(histogram)
                    results['azimuth_deg'] = float(peak_bin * (360.0 / self.K))
                    results['azimuth_bin'] = int(peak_bin)
                    results['confidence'] = float(histogram[peak_bin])
                    results['sources'] = [{
                        'azimuth_deg': results['azimuth_deg'],
                        'score': results['confidence'],
                        'prominence': results['confidence']
                    }]
                    results['num_sources'] = 1
            else:
                # No azimuth detected (all probabilities below threshold)
                results['sources'] = []
                results['num_sources'] = 0
                results['azimuth_deg'] = None
                results['azimuth_bin'] = None
                results['confidence'] = 0.0
            t_detect = time.perf_counter() - t_detect_start
            
            t_total = time.perf_counter() - t_total_start
            
            # Store timing information
            results['timing'] = {
                'total': t_total * 1000,  # ms
                'features': t_feat * 1000,
                'model': t_model * 1000,
                'histogram': t_hist * 1000,
                'detect': t_detect * 1000,
            }
            
            return results
    
    def plot_histogram(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        show: bool = True,
        title: Optional[str] = None
    ):
        """
        Plot the histogram of DOA predictions.
        
        Args:
            results: Results dictionary from inference() with 'histogram' and 'histogram_bins'
            save_path: Optional path to save the plot
            show: Whether to display the plot
            title: Optional title for the plot
        """
        if 'histogram' not in results:
            raise ValueError("Results dictionary must contain 'histogram' key. Run inference with return_histogram=True")
        
        histogram = results['histogram']
        bin_centers = results.get('histogram_bins', np.arange(len(histogram)) * (360.0 / len(histogram)))
        predicted_azimuth = results.get('azimuth_deg')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histogram as bar chart
        ax.bar(bin_centers, histogram, width=360.0/len(histogram), 
               edgecolor='black', linewidth=0.5, alpha=0.7, color='steelblue', 
               label='Histogram')
        
        # Mark predicted azimuth(s) if available
        sources = results.get('sources', [])
        if predicted_azimuth is not None:
            ax.axvline(x=predicted_azimuth, color='red', linestyle='--', linewidth=2, 
                      label=f"Primary: {predicted_azimuth:.1f}°")
        
        # Mark multiple sources if detected
        colors = ['green', 'orange', 'purple', 'brown', 'pink']
        for i, source in enumerate(sources):
            color = colors[i % len(colors)]
            ax.axvline(x=source['azimuth_deg'], color=color, linestyle='--', linewidth=2, 
                      alpha=0.7, label=f"Source {i+1}: {source['azimuth_deg']:.1f}° (score={source['score']:.3f})")
        
        # Formatting
        ax.set_xlabel('Azimuth (degrees)', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(title or 'DOA Prediction Histogram', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 360)
        ax.set_ylim(0, max(histogram) * 1.1 if histogram.max() > 0 else 0.1)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='upper right')
        
        # Add text with statistics
        stats_lines = []
        if predicted_azimuth is not None:
            confidence = results.get('confidence', 0.0)
            stats_lines.append(f"Primary: {predicted_azimuth:.1f}°")
            stats_lines.append(f"Confidence: {confidence:.3f}")
        
        num_sources = results.get('num_sources', 1)
        if num_sources > 1:
            stats_lines.append(f"Detected sources: {num_sources}")
            for i, source in enumerate(sources):
                stats_lines.append(f"  Source {i+1}: {source['azimuth_deg']:.1f}° (prom={source['prominence']:.3f})")
        
        if stats_lines:
            stats_text = "\n".join(stats_lines)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Histogram saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_per_frame_outputs(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        show: bool = True,
        max_frames: int = 100
    ):
        """
        Plot interactive per-frame DOA predictions.
        
        Args:
            results: Results dictionary from inference() with return_per_frame=True
            save_path: Optional path to save the plot
            show: Whether to display the plot
            max_frames: Maximum number of frames to display (for performance)
        """
        if 'per_frame_azimuth' not in results:
            raise ValueError("Results dictionary must contain 'per_frame_azimuth' key. Run inference with return_per_frame=True")
        
        per_frame_azimuth = results['per_frame_azimuth']
        per_frame_confidence = results.get('per_frame_confidence', np.zeros_like(per_frame_azimuth))
        vad_probs = results.get('vad_probs', np.ones_like(per_frame_azimuth))
        frame_times = results.get('frame_times', np.arange(len(per_frame_azimuth)) * 0.010)
        per_frame_probs = results.get('per_frame_probs', None)  # (T, K)
        
        T = len(per_frame_azimuth)
        
        # Limit frames for performance
        if T > max_frames:
            step = T // max_frames
            indices = np.arange(0, T, step)
            per_frame_azimuth = per_frame_azimuth[indices]
            per_frame_confidence = per_frame_confidence[indices]
            vad_probs = vad_probs[indices]
            frame_times = frame_times[indices]
            if per_frame_probs is not None:
                per_frame_probs = per_frame_probs[indices]
            T = len(indices)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Azimuth over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(frame_times, per_frame_azimuth, 'b-', linewidth=1.5, alpha=0.7, label='Predicted Azimuth')
        ax1.fill_between(frame_times, per_frame_azimuth - 5, per_frame_azimuth + 5, alpha=0.2, color='blue')
        ax1.set_xlabel('Time (seconds)', fontsize=11)
        ax1.set_ylabel('Azimuth (degrees)', fontsize=11)
        ax1.set_title('DOA Prediction Over Time', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 360)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Confidence over time
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(frame_times, per_frame_confidence, 'g-', linewidth=1.5, alpha=0.7, label='Confidence')
        ax2.set_xlabel('Time (seconds)', fontsize=11)
        ax2.set_ylabel('Confidence', fontsize=11)
        ax2.set_title('Prediction Confidence Over Time', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. VAD probabilities over time
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(frame_times, vad_probs, 'r-', linewidth=1.5, alpha=0.7, label='VAD Probability')
        vad_threshold = self.vad_threshold
        ax3.axhline(y=vad_threshold, color='orange', linestyle='--', linewidth=1, label=f'VAD Threshold ({vad_threshold})')
        ax3.fill_between(frame_times, 0, vad_probs, alpha=0.3, color='red', where=(vad_probs >= vad_threshold))
        ax3.set_xlabel('Time (seconds)', fontsize=11)
        ax3.set_ylabel('VAD Probability', fontsize=11)
        ax3.set_title('Voice Activity Detection', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Probability heatmap (if available)
        if per_frame_probs is not None:
            ax4 = fig.add_subplot(gs[2, :])
            
            # Create heatmap: time (x-axis) vs azimuth (y-axis)
            # per_frame_probs is (T, K), we transpose to (K, T) for imshow
            im = ax4.imshow(per_frame_probs.T, aspect='auto', origin='lower', 
                           cmap='viridis', interpolation='bilinear',
                           extent=[frame_times[0], frame_times[-1], 0, 360])
            
            # Overlay predicted azimuth line
            ax4.plot(frame_times, per_frame_azimuth, 'r-', linewidth=2, alpha=0.8, label='Predicted Azimuth')
            
            ax4.set_xlabel('Time (seconds)', fontsize=11)
            ax4.set_ylabel('Azimuth (degrees)', fontsize=11)
            ax4.set_title('Probability Distribution Over Time (Heatmap)', fontsize=12, fontweight='bold')
            ax4.set_ylim(0, 360)
            ax4.legend(loc='upper right')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label('Probability', fontsize=10)
        else:
            # If no probabilities, show azimuth scatter with confidence as color
            ax4 = fig.add_subplot(gs[2, :])
            scatter = ax4.scatter(frame_times, per_frame_azimuth, c=per_frame_confidence, 
                                cmap='viridis', s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax4.set_xlabel('Time (seconds)', fontsize=11)
            ax4.set_ylabel('Azimuth (degrees)', fontsize=11)
            ax4.set_title('Azimuth vs Time (colored by confidence)', fontsize=12, fontweight='bold')
            ax4.set_ylim(0, 360)
            ax4.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('Confidence', fontsize=10)
        
        plt.suptitle('Per-Frame DOA Analysis', fontsize=14, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Per-frame plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


# Example usage
if __name__ == "__main__":
    
    # Option 1: Load from checkpoint
    from pathlib import Path
    import yaml
    
    checkpoint_path = "models/basic/2025-11-06_22-37-00-6a5fbc92/last.pt"
    config_path = Path(__file__).parent.parent / "configs" / "train.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # model_name can be omitted - it will be auto-detected from checkpoint keys
    # Or explicitly specify: model_name='doa', 'basic', 'scat', 'film', 'retin'
    model = InferenceModel(
        checkpoint_path=checkpoint_path,
        config=config
        # model_name='doa'  # Optional: auto-detected if not provided
    )
    
    # # Option 2: Use model instance directly
    # from mirokai_doa.model_basic import TFPoolClassifierNoCond
    
    # model_instance = TFPoolClassifierNoCond(K=72)
    # model = InferenceModel(model=model_instance, config=config)
    
    # # Load test audio and run inference
    import soundfile as sf
    
    audio_path = Path(__file__).parent / "mixture_4ch.wav"
    if not audio_path.exists():
        # Try alternative file
        audio_path = Path(__file__).parent / "sample_mix.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found. Tried: mixture_4ch.wav and sample_mix.wav in {Path(__file__).parent}")
    
    print(f"Loading audio from: {audio_path}")
    mixture, sr = sf.read(audio_path)
    if mixture.ndim == 1:
         mixture = np.tile(mixture[:, None], (1, 4))  # Mono to 4ch
    mixture = mixture.T  # (4, T)
    
    # Run inference
    results = model.inference(mixture, return_histogram=True, return_per_frame=True)
    
    print(f"Predicted azimuth: {results['azimuth_deg']:.1f}°")
    print(f"Confidence: {results['confidence']:.3f}")
    print(f"Histogram shape: {results['histogram'].shape}")
    
    # Print detected sources
    if 'sources' in results and len(results['sources']) > 0:
        print(f"\nDetected {results['num_sources']} source(s):")
        for i, source in enumerate(results['sources']):
            print(f"  Source {i+1}: {source['azimuth_deg']:.1f}° "
                  f"(score={source['score']:.3f}, prominence={source['prominence']:.3f})")
    else:
        print("\nSingle source detected")
    
    # Plot histogram
    model.plot_histogram(
        results,
        save_path=Path(__file__).parent / "doa_histogram.png",
        show=True,
        title=f"DOA Histogram - Predicted: {results['azimuth_deg']:.1f}°"
    )
    
    # Plot per-frame outputs (interactive visualization)
    model.plot_per_frame_outputs(
        results,
        save_path=Path(__file__).parent / "doa_per_frame.png",
        show=True,
        max_frames=200  # Adjust based on your audio length
    )
