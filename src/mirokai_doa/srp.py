import numpy as np
from itertools import combinations
from scipy.signal import find_peaks

# ========================= Utilities =========================

def _unit_vectors_from_azimuths_deg(az_deg):
    th = np.deg2rad(np.asarray(az_deg))
    return np.stack([np.cos(th), np.sin(th)], axis=-1)  # (A,2)

def _pair_list(M):
    return list(combinations(range(M), 2))

def _precompute_tdoa_seconds(mic_xy, az_deg, c=343.0):
    """
    Predicted signed TDOAs in seconds for every azimuth and mic pair (i<j).
    τ(θ) = -(Δr · u)/c, with Δr = r_j - r_i and u = [cosθ, sinθ].
    Returns:
      tau_sec: (A, P) float
      pairs:   list[(i,j)] length P
    """
    mic_xy = np.asarray(mic_xy, dtype=float)   # (M,2)
    A = _unit_vectors_from_azimuths_deg(az_deg)  # (A,2)
    pairs = _pair_list(mic_xy.shape[0])
    dr = np.stack([mic_xy[j] - mic_xy[i] for (i, j) in pairs], axis=0)  # (P,2)
    tau_sec = - A @ dr.T / c  # (A,P)
    return tau_sec, pairs

def _freq_mask(freqs, fmin, fmax):
    if fmin is None and fmax is None:
        return np.ones_like(freqs, dtype=bool)
    fmin = -np.inf if fmin is None else float(fmin)
    fmax = +np.inf if fmax is None else float(fmax)
    return (freqs >= fmin) & (freqs <= fmax)

# ========================= GCC-PHAT / SRP-PHAT from STFT =========================

def gcc_phat_azimuth_map_360_from_stft(
    X, freqs, mic_xy, *,
    az_res_deg=1.0,
    frame_weights=None,     # (T,) or None
    phat=True,              # PHAT-normalize per frame (recommended)
    drop_dc=True,           # zero DC bin contribution
    fmin=None, fmax=None,   # optional band limiting in Hz
    c=343.0, eps=1e-10
):
    """
    GCC/SRP-PHAT steered response map over 0..359° from STFT.
    X:      (T, M, F) complex STFT
    freqs:  (F,) frequency bins in Hz
    mic_xy: (M,2) mic positions in meters (XY)
    Returns: az_deg (A,), score (A,)
    """
    X = np.asarray(X)
    freqs = np.asarray(freqs, dtype=float)
    assert X.ndim == 3, "X must be (T, M, F)"
    T, M, F = X.shape
    assert freqs.shape[0] == F, "freqs must match X.shape[-1]"

    az_deg = np.arange(0.0, 360.0, az_res_deg)
    A = az_deg.size
    tau_sec, pairs = _precompute_tdoa_seconds(mic_xy, az_deg, c=c)

    # Frequency selection (optional)
    fmask = _freq_mask(freqs, fmin, fmax)
    if drop_dc and F > 0:
        fmask = fmask.copy()
        fmask[0] = False
    freqs_sel = freqs[fmask]                            # (F_sel,)
    X_sel = X[..., fmask]                               # (T, M, F_sel)
    F_sel = freqs_sel.shape[0]
    if F_sel == 0:
        raise ValueError("No frequency bins selected after applying fmin/fmax/DC drop.")

    # Optional frame weights
    if frame_weights is not None:
        w = np.asarray(frame_weights, dtype=float)
        if w.ndim != 1 or w.shape[0] != T:
            raise ValueError("frame_weights must be shape (T,)")
        w_sum = w.sum() + 1e-12  # avoid zero-div
        w = w[:, None]  # (T,1) for broadcasting over F_sel
    else:
        w = None
        w_sum = float(T)

    # PHAT-averaged cross spectra per pair (F_sel,)
    C_pairs = []
    for (i, j) in pairs:
        C = X_sel[:, i, :] * np.conj(X_sel[:, j, :])    # (T, F_sel)
        if phat:
            C = C / (np.abs(C) + eps)                   # PHAT per frame
        if w is not None:
            C_avg = (w * C).sum(axis=0) / w_sum
        else:
            C_avg = C.mean(axis=0)
        C_pairs.append(C_avg)                           # list of (F_sel,)

    # Steering & accumulation
    score = np.zeros(A, dtype=float)
    two_pi = 2.0 * np.pi
    for p, C_avg in enumerate(C_pairs):
        # phase[a,f] = exp(-j 2π f τ[a,p])
        phase = np.exp(-1j * two_pi * np.outer(tau_sec[:, p], freqs_sel))  # (A,F_sel)
        contrib = np.real(phase @ C_avg)                                    # (A,)
        score += contrib

    # Normalize to [0,1]
    score -= np.min(score)
    if np.max(score) > 0:
        score /= np.max(score)

    return az_deg, score

def srp_phat_azimuth_map_360_from_stft(
    X, freqs, mic_xy, az_res_deg=1.0, c=343.0, eps=1e-10
):
    """
    Same computation as your original srp_phat_azimuth_map_360, but takes X (T,M,F) and freqs (F,).
    Steps preserved:
      - C = X[:,i,:] * conj(X[:,j,:])
      - PHAT per frame: C /= (|C| + eps)
      - Mean over frames (unweighted)
      - Set DC bin to 0
      - Phase steering with exp(-j 2π f τ_ij(θ)), real-part sum over freqs and pairs
      - Normalize to [0,1]
    """
    X = np.asarray(X)
    freqs = np.asarray(freqs, dtype=float)
    T, M, F = X.shape
    az_deg = np.arange(0.0, 360.0, az_res_deg)
    A = az_deg.size

    # Precompute TDOAs (seconds) for all azimuths & pairs
    def _unit_vectors_from_azimuths_deg(az):
        th = np.deg2rad(np.asarray(az))
        return np.stack([np.cos(th), np.sin(th)], axis=-1)

    from itertools import combinations
    def _pair_list(M): return list(combinations(range(M), 2))

    U = _unit_vectors_from_azimuths_deg(az_deg)  # (A,2)
    pairs = _pair_list(M)                        # list of (i,j)
    dr = np.stack([mic_xy[j] - mic_xy[i] for (i, j) in pairs], axis=0)  # (P,2)
    tau_sec = - U @ dr.T / c                    # (A,P)

    # Per-pair averaged PHAT cross-spectra (F,)
    C_pairs = []
    for (i, j) in pairs:
        C = X[:, i, :] * np.conj(X[:, j, :])      # (T, F)
        C = C / (np.abs(C) + eps)                 # PHAT per frame
        C_avg = C.mean(axis=0)                    # (F,)
        # zero DC like your code
        if F > 0:
            C_avg = C_avg.copy()
            C_avg[0] = 0.0 + 0.0j
        C_pairs.append(C_avg)

    # Steering & accumulation (exactly as before)
    score = np.zeros(A, dtype=float)
    two_pi = 2.0 * np.pi
    for p, C_avg in enumerate(C_pairs):
        phase = np.exp(-1j * two_pi * np.outer(tau_sec[:, p], freqs))  # (A,F)
        contrib = np.real(phase @ C_avg)                                # (A,)
        score += contrib

    # Normalize to [0,1] like before
    score -= np.min(score)
    if np.max(score) > 0:
        score /= np.max(score)

    return az_deg, score

def srp_weighted_VAD_phat_azimuth_map_360_from_stft(
    X, freqs, mic_xy, vad_weights, az_res_deg=1.0, c=343.0, eps=1e-10
):
    """
    Weighted variant matching your previous code order:
      C = (vad * C)  -> PHAT per frame -> mean over frames
    (Note: mathematically, many do PHAT first then weight; but we keep your order to preserve results.)
    """
    X = np.asarray(X)
    freqs = np.asarray(freqs, dtype=float)
    vad = np.asarray(vad_weights, dtype=float)
    T, M, F = X.shape
    assert vad.shape == (T,), "vad_weights must be shape (T,)"

    az_deg = np.arange(0.0, 360.0, az_res_deg)
    A = az_deg.size

    # TDOAs
    def _unit_vectors_from_azimuths_deg(az):
        th = np.deg2rad(np.asarray(az))
        return np.stack([np.cos(th), np.sin(th)], axis=-1)

    from itertools import combinations
    def _pair_list(M): return list(combinations(range(M), 2))

    U = _unit_vectors_from_azimuths_deg(az_deg)  # (A,2)
    pairs = _pair_list(M)
    dr = np.stack([mic_xy[j] - mic_xy[i] for (i, j) in pairs], axis=0)  # (P,2)
    tau_sec = - U @ dr.T / c                                            # (A,P)

    # Per-pair weighted PHAT cross-spectra (preserving your order)
    C_pairs = []
    w = vad[:, None]  # (T,1)
    wsum = vad.sum() + 1e-12
    for (i, j) in pairs:
        C = X[:, i, :] * np.conj(X[:, j, :])      # (T, F)
        C = w * C                                 # apply frame weights first (as you had)
        C = C / (np.abs(C) + eps)                 # PHAT per frame
        C_avg = C.sum(axis=0) / wsum              # weighted mean across frames
        if F > 0:
            C_avg = C_avg.copy()
            C_avg[0] = 0.0 + 0.0j                 # zero DC
        C_pairs.append(C_avg)

    # Steering & accumulation (same as before)
    score = np.zeros(A, dtype=float)
    two_pi = 2.0 * np.pi
    for p, C_avg in enumerate(C_pairs):
        phase = np.exp(-1j * two_pi * np.outer(tau_sec[:, p], freqs))  # (A,F)
        contrib = np.real(phase @ C_avg)                                # (A,)
        score += contrib

    score -= np.min(score)
    if np.max(score) > 0:
        score /= np.max(score)

    return az_deg, score
# ========================= Peak picker (unchanged) =========================

def pick_peaks_360_robust(
    az_deg, score, top_k=5, min_separation_deg=10.0, min_prominence=0.05,
    refine_subdegree=True
):
    """
    Robust circular peak picker (0..360 wrap-aware).
    - Uses scipy.signal.find_peaks on a triplicated signal to enforce wrap-around separation.
    - Ranks by *prominence*, then by height.
    - Optional sub-degree parabolic refinement (±1 bin).
    Returns: list of dicts: [{'azimuth_deg': float, 'score': float, 'prominence': float}]
    """
    az = np.asarray(az_deg)
    s = np.asarray(score)
    assert az.ndim == 1 and s.ndim == 1 and az.size == s.size
    n = s.size
    if n < 3:
        return []

    # Triplicate to handle circularity
    s_ext = np.concatenate([s, s, s])
    # Use bin distance corresponding to min_separation_deg
    bin_deg = np.mean(np.diff(az)) if n > 1 else 360.0
    min_dist_bins = max(1, int(np.round(min_separation_deg / bin_deg)))

    # Prominence threshold as absolute value (relative to peak=1 if map is normalized)
    prom_abs = float(min_prominence)  # assume score normalized in [0,1]; adjust if not

    peaks, props = find_peaks(s_ext, distance=min_dist_bins, prominence=prom_abs)
    if peaks.size == 0:
        return []

    # Keep only peaks in the central copy
    mid_start, mid_end = n, 2*n
    mask_mid = (peaks >= mid_start) & (peaks < mid_end)
    peaks = peaks[mask_mid]
    if peaks.size == 0:
        return []

    prominences = props["prominences"][mask_mid]
    heights = s_ext[peaks]

    # Rank by prominence then height
    order = np.lexsort((-heights, -prominences))  # descending
    peaks = peaks[order]
    prominences = prominences[order]

    results = []
    for k in range(min(top_k, peaks.size)):
        idx_ext = peaks[k]
        idx = idx_ext - n  # map back to original [0..n-1]
        # optional sub-degree refinement via quadratic fit
        if refine_subdegree and 1 <= idx < n-1:
            y1, y2, y3 = s[idx-1], s[idx], s[idx+1]
            denom = (y1 - 2*y2 + y3)
            if abs(denom) > 1e-12:
                delta = 0.5 * (y1 - y3) / denom  # offset in bins within (-0.5,0.5)
            else:
                delta = 0.0
        else:
            delta = 0.0

        az_refined = (az[idx] + delta * bin_deg) % 360.0
        results.append({
            "azimuth_deg": float(az_refined),
            "score": float(s[idx]),
            "prominence": float(prominences[k]),
            "index": int(idx),
        })

    return results


# ========================= Wrapper (now takes STFT) =========================

def estimate_azimuths_360_from_stft(
    X, freqs, mic_xy, *,
    method="srp",
    az_res_deg=1.0,
    peak_kwargs=None,
    **map_kwargs
):
    """
    Compute a 0..360° map (SRP/GCC) from STFT and return up to 5 DOA candidates.
    X: (T, M, F), freqs: (F,), mic_xy: (M,2)
    method: 'srp' or 'gcc'
    map_kwargs: forwarded to *_from_stft functions (e.g., frame_weights, fmin/fmax, drop_dc)
    Returns: {'azimuths': list[dict], 'az_grid_deg': np.ndarray, 'map': np.ndarray, 'method': str}
    """
    peak_kwargs = peak_kwargs or {}
    method = method.lower()
    if method == "srp":
        az, s = srp_phat_azimuth_map_360_from_stft(X, freqs, mic_xy, az_res_deg=az_res_deg, **map_kwargs)
    elif method == "gcc":
        az, s = gcc_phat_azimuth_map_360_from_stft(X, freqs, mic_xy, az_res_deg=az_res_deg, **map_kwargs)
    else:
        raise ValueError("method must be 'srp' or 'gcc'")

    peaks = pick_peaks_360_robust(az, s, **peak_kwargs)
    return {"azimuths": peaks, "az_grid_deg": az, "map": s, "method": method}

