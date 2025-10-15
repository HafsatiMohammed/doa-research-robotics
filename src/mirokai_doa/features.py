import numpy as np
from numpy.fft import rfft
from numpy.lib.stride_tricks import as_strided
from scipy.signal import get_window

def stft_multi(
    x,
    fs: float,
    win_s: float = 0.032,
    hop_s: float = 0.010,
    nfft: int | None = None,
    window: str | tuple | np.ndarray = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    out_dtype = np.complex64,
):
    """
    Multichannel STFT (vectorized).
    Args
    ----
    x       : np.ndarray, shape (N, C)  time-domain signal
    fs      : float, sampling rate (Hz)
    win_s   : float, window length in seconds (default 32 ms)
    hop_s   : float, hop length in seconds (default 10 ms)
    nfft    : int or None. If None, uses next power of two >= frame_len
    window  : str/tuple/array for scipy.signal.get_window or a length-L array
    center  : if True, pad by L//2 on both sides (librosa-style)
    pad_mode: np.pad mode (e.g., "reflect", "constant")
    out_dtype: dtype for STFT output (complex64 recommended)

    Returns
    -------
    X   : np.ndarray, shape (T, C, F) complex STFT
    freqs: np.ndarray, shape (F,) frequency bins in Hz
    times: np.ndarray, shape (T,) frame center times in seconds
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]  # (N,1)
    assert x.ndim == 2, "x must be (samples, channels)"
    N, C = x.shape

    # Window & hop in samples
    frame_len = int(round(win_s * fs))
    hop = int(round(hop_s * fs))
    if frame_len <= 0 or hop <= 0:
        raise ValueError("win_s and hop_s must be > 0")

    # FFT size
    def _next_pow2(n):
        return 1 << (int(n - 1).bit_length())
    nfft = _next_pow2(frame_len) if nfft is None else int(nfft)
    if nfft < frame_len:
        raise ValueError("nfft must be >= frame_len")

    # Window vector
    if isinstance(window, np.ndarray):
        w = window.astype(float, copy=False)
    else:
        w = get_window(window, frame_len, fftbins=True).astype(float)
    if w.shape[0] != frame_len:
        raise ValueError("Provided window length != frame_len")

    # Optional centering (pad by L//2 on both sides)
    pad = frame_len // 2 if center else 0
    if pad > 0:
        x_pad = np.pad(x, ((pad, pad), (0, 0)), mode=pad_mode)
    else:
        x_pad = x

    Np = x_pad.shape[0]
    if Np < frame_len:
        # ensure at least one frame
        x_pad = np.pad(x_pad, ((0, frame_len - Np), (0, 0)), mode=pad_mode)
        Np = x_pad.shape[0]

    # Number of frames
    T = 1 + (Np - frame_len) // hop
    if T <= 0:
        raise ValueError("Signal too short for given window/hop")

    # Stride-trick framing: (T, frame_len, C) view into x_pad
    s_t, s_c = x_pad.strides  # bytes per step in time/channel
    frames = as_strided(
        x_pad,
        shape=(T, frame_len, C),
        strides=(hop * s_t, s_t, s_c),
        writeable=False,
    )
    # Reorder to (T, C, frame_len) to apply window & FFT along the last axis
    frames = np.transpose(frames, (0, 2, 1))  # (T, C, L)

    # Apply window (broadcast over T and C)
    frames = frames * w[None, None, :]

    # Batched real FFT along last axis -> (T, C, F)
    X = rfft(frames, n=nfft, axis=-1).astype(out_dtype, copy=False)

    # Frequency and time vectors
    F = X.shape[-1]
    freqs = (fs / nfft) * np.arange(F)
    # Frame centers relative to original signal
    if center:
        # centers at sample indices: t*hop  (librosa convention)
        times = (np.arange(T) * hop) / fs
    else:
        # window centered at (frame_len/2) + t*hop
        times = (np.arange(T) * hop + frame_len / 2.0) / fs

    return X, freqs, times



def _wrap_to_2pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles to [0, 2π)."""
    return np.mod(x, 2.0 * np.pi)

def compute_mag_phase(
    X: np.ndarray,
    dtype=np.float32,
):
    """
    Per-channel magnitude and absolute phase (wrapped to [0, 2π)).

    Args
    ----
    X    : np.ndarray, shape (T, C, F), complex STFT
    dtype: output dtype

    Returns
    -------
    mag   : np.ndarray, shape (T, C, F) = |X|
    phase : np.ndarray, shape (T, C, F) = angle(X) in [0, 2π)
    """
    assert X.ndim == 3, "X must be (T, C, F)"
    mag = np.abs(X).astype(dtype, copy=False)
    phase = _wrap_to_2pi(np.angle(X)).astype(dtype, copy=False)
    return mag, phase

def compute_mag_phase_cos_sin(
    X: np.ndarray,
    dtype=np.float32,
):
    """
    Concatenate per-channel magnitude, cos(phase), sin(phase).

    Args
    ----
    X    : np.ndarray, shape (T, C, F), complex STFT
    dtype: output dtype

    Returns
    -------
    feats : np.ndarray, shape (T, 3*C, F)
        Layout = [mag (C), cos(phase) (C), sin(phase) (C)]
        where phase is angle(X) wrapped to [0, 2π).
    """
    mag, phase = compute_mag_phase(X, dtype=dtype)
    cos_phase = np.cos(phase).astype(dtype, copy=False)
    sin_phase = np.sin(phase).astype(dtype, copy=False)
    feats = np.concatenate([mag, cos_phase, sin_phase], axis=1)
    return feats

def compute_real_imag_features(
    X: np.ndarray,
    dtype=np.float32,
):
    """
    Concatenate per-channel real and imaginary parts.

    Args
    ----
    X    : np.ndarray, shape (T, C, F), complex STFT
    dtype: output dtype

    Returns
    -------
    feats : np.ndarray, shape (T, 2*C, F)
        Layout = [Re (C), Im (C)]
    """
    assert X.ndim == 3, "X must be (T, C, F)"
    real = X.real.astype(dtype, copy=False)
    imag = X.imag.astype(dtype, copy=False)
    feats = np.concatenate([real, imag], axis=1)
    return feats

