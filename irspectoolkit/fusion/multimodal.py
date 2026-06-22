"""多光谱融合 — NIR+MIR, IR+Raman"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d


def align_wavelengths(
    spectra_list: list[np.ndarray],
    wavelength_list: list[np.ndarray],
    target_wavelengths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将多条光谱对齐到统一波长网格
    
    通过插值实现
    """
    if target_wavelengths is None:
        all_wl = np.concatenate(wavelength_list)
        target_wavelengths = np.sort(np.unique(all_wl))

    aligned = []
    for spec, wl in zip(spectra_list, wavelength_list):
        if spec.ndim == 1:
            spec = spec.reshape(1, -1)
        for s in spec:
            f = interp1d(wl, s, kind="linear", fill_value="extrapolate")
            aligned.append(f(target_wavelengths))

    return np.array(aligned), target_wavelengths


def early_fusion(
    spectra_list: list[np.ndarray],
    wavelength_list: list[np.ndarray] | None = None,
) -> np.ndarray:
    """
    早期融合 — 直接拼接
    
    对齐波长后水平拼接所有光谱
    """
    if wavelength_list is not None:
        aligned, _ = align_wavelengths(spectra_list, wavelength_list)
        return aligned
    else:
        return np.hstack(spectra_list)


def late_fusion(
    predictions_list: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """
    晚期融合 — 加权投票/平均
    
    对各模态的预测结果进行加权融合
    """
    if weights is None:
        weights = [1.0 / len(predictions_list)] * len(predictions_list)

    predictions = np.zeros_like(predictions_list[0], dtype=float)
    for pred, w in zip(predictions_list, weights):
        predictions += w * pred.astype(float)

    return predictions
