from scipy.stats import zscore
import numpy as np
from typing import Sequence
from sklearn.preprocessing import StandardScaler
# 이상치 탐지

def compute_zscore_scipy(data: Sequence[float]) -> np.ndarray:
    return zscore(data)



def compute_zscore_sklearn(data: Sequence[float]) -> np.ndarray:
    arr = np.array(data, dtype=float).reshape(-1, 1) 
    scaler = StandardScaler()
    result = scaler.fit_transform(arr)
    return result.flatten()

def compute_zscore_custom(data: Sequence[float]) -> list[float]:
    mean = sum(data) / len(data)
    std = (sum((x - mean)**2 for x in data) / len(data))**0.5
    return [(x - mean) / std for x in data]


# 이상치 제거



def remove_outliers_zscore(data: Sequence[float], threshold: float = 3.0) -> np.ndarray:
    data_array = np.array(data)
    z_scores = zscore(data_array)
    mask = np.abs(z_scores) < threshold
    return data_array[mask]
