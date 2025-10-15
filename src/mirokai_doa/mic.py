import math
import numpy as np
from dataclasses import dataclass



@dataclass
class Params:
    c: float = 343.0
    radius_mm: float = 27.7
    mic_angles_deg: tuple = (0.0, 90.0, 180.0, 270.0)
    fs: int = 16000
    frame_ms: float = 32.0
    hop_ms: float = 10.0
    nfft: int = 1024
    beta: float = 1.0
    az_deg_res: float = 1.0
    grid_extent_m: float = 0.60
    grid_points: int = 201
    topk: int = 3
    eps: float = 1e-12
    softmax_temp: float = 0.1  # temperature for probability over azimuth



def ring_geometry_meters(params: Params):
    r = params.radius_mm * 1e-3
    xy = []
    for ang in params.mic_angles_deg:
        rad = math.radians(ang)
        xy.append([r * math.cos(rad), r * math.sin(rad)])
    return np.array(xy, dtype=np.float64)
    
    
    


if __name__ == "__main__":

	P = Params(fs=16000)
	mic_xy = ring_geometry_meters(P)
    
    
    
    
