"""
光谱预处理变换 — SNV, MSC, 导数, 平滑, 归一化
全部基于numpy/scipy实现，兼容sklearn Pipeline
"""
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# SNV (Standard Normal Variate) — 标准正态变换
# ============================================================
def snv(spectra: np.ndarray) -> np.ndarray:
    """
    标准正态变换: 每条光谱减去自身均值，除以自身标准差
    
    Args:
        spectra: (n_samples, n_wavelengths)
    Returns:
        变换后的光谱
    """
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (spectra - mean) / std


class SNVTransformer(BaseEstimator, TransformerMixin):
    """sklearn兼容的SNV变换器"""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return snv(X)


# ============================================================
# MSC (Multiplicative Scatter Correction) — 多元散射校正
# ============================================================
def msc(spectra: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """
    多元散射校正
    
    以平均光谱为参考，对每条光谱做线性回归修正散射效应
    X_corrected = (X - intercept) / slope
    """
    if reference is None:
        reference = spectra.mean(axis=0)

    corrected = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        # 线性回归: spectra[i] = slope * reference + intercept
        slope, intercept = np.polyfit(reference, spectra[i], 1)
        if slope == 0:
            slope = 1.0
        corrected[i] = (spectra[i] - intercept) / slope

    return corrected


class MSCTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return msc(X)


# ============================================================
# Savitzky-Golay 平滑
# ============================================================
def savgol_smooth(
    spectra: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    """Savitzky-Golay平滑"""
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(spectra, window_length, polyorder, axis=1, mode="nearest")


# ============================================================
# Savitzky-Golay 导数
# ============================================================
def savgol_derivative(
    spectra: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
    deriv: int = 1,
) -> np.ndarray:
    """
    Savitzky-Golay一阶/二阶导数
    
    Args:
        deriv: 导数阶数 (1=一阶, 2=二阶)
    """
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(
        spectra, window_length, polyorder, deriv=deriv, axis=1, mode="nearest"
    )


# ============================================================
# Min-Max 归一化
# ============================================================
def minmax_normalize(spectra: np.ndarray) -> np.ndarray:
    """每条光谱归一化到[0, 1]"""
    min_val = spectra.min(axis=1, keepdims=True)
    max_val = spectra.max(axis=1, keepdims=True)
    denom = max_val - min_val
    denom[denom == 0] = 1.0
    return (spectra - min_val) / denom


# ============================================================
# 波长区间裁剪
# ============================================================
def wavelength_cut(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    w_min: float | None = None,
    w_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """裁剪到指定波长区间"""
    mask = np.ones(len(wavelengths), dtype=bool)
    if w_min is not None:
        mask &= wavelengths >= w_min
    if w_max is not None:
        mask &= wavelengths <= w_max
    return spectra[:, mask], wavelengths[mask]
