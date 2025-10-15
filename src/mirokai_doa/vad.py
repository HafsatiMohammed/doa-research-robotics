# original: https://github.com/snakers4/silero-vad/blob/master/utils_vad.py

import logging
import os
import subprocess

import numpy as np
import onnxruntime
import torch

RATE = 16000

# Create a specific logger for voice activity detection
Logger = logging.getLogger("voice_activity_detector")


class VoiceActivityDetection:

    def __init__(self, force_onnx_cpu=True):
        Logger.info("Initializing VoiceActivityDetection model")
        path = self.download()
        opts = onnxruntime.SessionOptions()
        opts.log_severity_level = 3

        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if force_onnx_cpu and "CPUExecutionProvider" in onnxruntime.get_available_providers():
            Logger.info("Using CPU execution provider for ONNX runtime")
            self.session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
        else:
            Logger.info("Using CUDA execution provider for ONNX runtime")
            self.session = onnxruntime.InferenceSession(path, providers=["CUDAExecutionProvider"], sess_options=opts)

        self.reset_states()
        self.sample_rates = [8000, RATE]
        Logger.info(f"VoiceActivityDetection model initialized with sample rates: {self.sample_rates}")

    def _validate_input(self, x, sr: int):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        if sr != RATE and (sr % RATE == 0):
            step = sr // RATE
            x = x[:, ::step]
            sr = RATE

        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of {RATE})")

        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size=1):
        self._h = np.zeros((2, batch_size, 64)).astype("float32")
        self._c = np.zeros((2, batch_size, 64)).astype("float32")
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int):

        x, sr = self._validate_input(x, sr)
        batch_size = x.shape[0]

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if sr in [8000, RATE]:  # Use global RATE instead of 16000
            ort_inputs = {"input": x.numpy(), "h": self._h, "c": self._c, "sr": np.array(sr, dtype="int64")}
            ort_outs = self.session.run(None, ort_inputs)
            out, self._h, self._c = ort_outs
        else:
            raise ValueError()

        self._last_sr = sr
        self._last_batch_size = batch_size

        out = torch.tensor(out)
        return out

    def audio_forward(self, x, sr: int, num_samples: int = 512):
        outs = []
        x, sr = self._validate_input(x, sr)

        if x.shape[1] % num_samples:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = torch.nn.functional.pad(x, (0, pad_num), "constant", value=0.0)

        self.reset_states(x.shape[0])
        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i : i + num_samples]
            out_chunk = self.__call__(wavs_batch, sr)
            outs.append(out_chunk)

        stacked = torch.cat(outs, dim=1)
        return stacked.cpu()

    @staticmethod
    def download(
        model_url="https://raw.githubusercontent.com/snakers4/silero-vad/1baf307b35ab3bbb070ab374b43a0a3c3604fa2a/files/silero_vad.onnx",
    ):
        target_dir = os.path.dirname(__file__)+ "/models/"

        # Ensure the target directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Define the target file path
        model_filename = os.path.join(target_dir, "silero_vad.onnx")

        # Check if the model file already exists
        if not os.path.exists(model_filename):
            Logger.info(f"Downloading VAD model to {model_filename}")
            # If it doesn't exist, download the model using wget
            try:
                subprocess.run(["wget", "-O", model_filename, model_url], check=True)
                Logger.info("VAD model downloaded successfully")
            except subprocess.CalledProcessError:
                Logger.error("Failed to download the VAD model using wget.")
        else:
            Logger.info(f"VAD model already exists at {model_filename}")
        return model_filename


class VoiceActivityDetector:
    def __init__(self, threshold=0.2, frame_rate=RATE):
        """
        Initializes the VoiceActivityDetector with a voice activity detection model and a threshold.

        Args:
            threshold (float, optional): The probability threshold for detecting voice activity. Defaults to 0.5.
        """
        Logger.info(f"Initializing VoiceActivityDetector with threshold: {threshold}, frame_rate: {frame_rate}")
        self.model = VoiceActivityDetection()
        self.threshold = threshold
        self.frame_rate = frame_rate
        Logger.info("VoiceActivityDetector initialization completed")

    def __call__(self, audio_frame):
        """
        Determines if the given audio frame contains speech by comparing the detected speech probability against
        the threshold.

        Args:
            audio_frame (np.ndarray): The audio frame to be analyzed for voice activity. It is expected to be a
                                      NumPy array of audio samples.

        Returns:
            bool: True if the speech probability exceeds the threshold, indicating the presence of voice activity;
                  False otherwise.
        """
        speech_prob = self.model(torch.from_numpy(audio_frame.copy()), self.frame_rate).item()
        return speech_prob > self.threshold
    


    def vad_probs_same_frame_count(
        self,
        audio: np.ndarray,
        sr: int,
        hop_ms: float = 10.0,       # your STFT hop
        base_win_ms: float = 32.0,  # your STFT window that defined the centers
        big_win_ms: float = 500.0,  # the larger VAD window you want
        n_frames: int | None = None,  # if you already know (e.g., times.shape[0] == 650)
        pad_mode: str = "constant",  # "constant" zeros or "reflect"
    ):
        """
        Keep the SAME frame centers as the STFT computed with center=False and hop=hop_ms,
        but evaluate VAD with a larger window around each center. Edge frames are padded.
        Returns exactly n_frames probabilities (same as your STFT).
        """
        # --- mono, float32, safe range
        if audio.ndim != 1:
            raise ValueError("Pass mono [samples] (use audio[:, ch] upstream).")
        a = audio.astype(np.float32, copy=False)
        maxabs = float(np.max(np.abs(a))) if a.size else 0.0
        if np.isfinite(maxabs) and maxabs > 1.0:
            a = a / maxabs
        a = np.clip(a, -1.0, 1.0)

        hop = int(round(hop_ms * sr / 1000.0))            # 160 @ 16k
        base_win = int(round(base_win_ms * sr / 1000.0))  # 512 @ 16k (your STFT)
        big_win = int(round(big_win_ms * sr / 1000.0))    # e.g., 8000 @ 16k

        if hop <= 0 or base_win <= 0 or big_win <= 0:
            raise ValueError("hop_ms/base_win_ms/big_win_ms must be > 0.")

        # --- number of frames and baseline centers (center=False)
        N = len(a)
        if n_frames is None:
            # STFT, center=False: starts = 0, hop, 2*hop, ...
            # n_frames = 1 + floor((N - base_win) / hop)  (>=1 if N>=base_win)
            n_frames = 1 + max(0, (N - base_win) // hop)
        # baseline centers in samples (center of each base window)
        centers = (base_win // 2) + np.arange(n_frames, dtype=np.int64) * hop

        # --- evaluate VAD on bigger window centered on the same centers
        half = big_win // 2
        probs = np.empty(n_frames, dtype=np.float32)

        for i, c in enumerate(centers):
            start = int(c - half)
            end   = int(start + big_win)

            # pad if the window goes out of bounds
            if start < 0 or end > N:
                # slice within bounds
                s0 = max(0, start)
                s1 = min(N, end)
                frame = a[s0:s1]
                left_pad  = s0 - start   # positive if start<0
                right_pad = end - s1     # positive if end>N
                if left_pad > 0 or right_pad > 0:
                    frame = np.pad(frame, (max(0,left_pad), max(0,right_pad)), mode=pad_mode)
            else:
                frame = a[start:end]

            # ensure exact length
            if frame.shape[0] != big_win:
                frame = np.pad(frame, (0, big_win - frame.shape[0]), mode=pad_mode)

            # independent decision per frame (reset state)
            self.model.reset_states(batch_size=1)
            with torch.no_grad():
                probs[i] = self.model(torch.from_numpy(frame[None, :]), sr).item()

        return probs