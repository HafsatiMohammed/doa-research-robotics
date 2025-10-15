import numpy as np

# -------- WAV loading (best-effort) --------
def load_wav_best_effort(path):
    try:
        import soundfile as sf
        data, fs = sf.read(path, always_2d=True)
        data = data.astype(np.float64)
        return data[:,:], fs
    except Exception as e_sf:
        try:
            from scipy.io import wavfile
            fs, data = wavfile.read(path)
            if data.ndim == 1:
                data = data[:, None]
            if np.issubdtype(data.dtype, np.integer):
                maxv = np.iinfo(data.dtype).max
                data = data.astype(np.float64) / maxv
            else:
                data = data.astype(np.float64)
            return data[:,:], fs
        except Exception as e_sp:
            raise RuntimeError(f"Could not read WAV. soundfile={e_sf}; scipy={e_sp}")


