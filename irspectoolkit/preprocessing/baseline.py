"""
基线校正方法 — 封装pybaselines + 原生实现
支持: AsLS, airPLS, SNIP, ModPoly, arPLS, DrPLS
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def baseline_als(
    spectra: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    niter: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Asymmetric Least Squares (AsLS) 基线校正
    
    Args:
        lam: 平滑参数 (越大越平滑, 推荐1e4-1e8)
        p: 非对称参数 (越小越拟合基线, 推荐0.001-0.05)
        niter: 迭代次数
    
    Returns:
        (corrected_spectra, baselines)
    """
    L = spectra.shape[1]
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    H = lam * D.dot(D.T)

    corrected = np.zeros_like(spectra)
    baselines = np.zeros_like(spectra)

    for i in range(spectra.shape[0]):
        y = spectra[i]
        w = np.ones(L)
        for _ in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + H
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        corrected[i] = y - z
        baselines[i] = z

    return corrected, baselines


def baseline_airpls(
    spectra: np.ndarray,
    lam: float = 1e4,
    niter: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    airPLS (Adaptive Iteratively Reweighted Penalized Least Squares)
    
    自适应调整权重，不需要手动设置p参数
    """
    L = spectra.shape[1]
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    H = lam * D.dot(D.T)

    corrected = np.zeros_like(spectra)
    baselines = np.zeros_like(spectra)

    for i in range(spectra.shape[0]):
        y = spectra[i]
        w = np.ones(L)
        for _ in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + H
            z = spsolve(Z, w * y)
            d = y - z
            d_neg = d[d < 0]
            if len(d_neg) == 0:
                break
            mean_neg = np.mean(np.abs(d_neg))
            std_neg = np.std(d_neg)
            threshold = -mean_neg - 2 * std_neg
            w = np.where(d < threshold, 0, np.exp(-(d - threshold)**2 / (2 * mean_neg**2 + 1e-10)))

        corrected[i] = y - z
        baselines[i] = z

    return corrected, baselines


def baseline_snip(
    spectra: np.ndarray,
    max_iterations: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping)
    
    快速且不需要调参，适合实时处理
    """
    corrected = np.zeros_like(spectra)
    baselines = np.zeros_like(spectra)

    for i in range(spectra.shape[0]):
        y = spectra[i].copy()
        for p in range(1, max_iterations + 1):
            for j in range(p, len(y) - p):
                avg = (y[j - p] + y[j + p]) / 2
                if y[j] > avg:
                    y[j] = avg
        corrected[i] = spectra[i] - y
        baselines[i] = y

    return corrected, baselines


def baseline_modpoly(
    spectra: np.ndarray,
    degree: int = 5,
    niter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Modified Polynomial Baseline
    
    迭代多项式拟合，逐步将峰区域权重降为0
    """
    corrected = np.zeros_like(spectra)
    baselines = np.zeros_like(spectra)

    for i in range(spectra.shape[0]):
        y = spectra[i].copy()
        weights = np.ones(len(y))

        for _ in range(niter):
            coeffs = np.polyfit(np.arange(len(y)), y, degree, w=weights)
            baseline = np.polyval(coeffs, np.arange(len(y)))
            residual = spectra[i] - baseline
            weights = np.where(residual < 0, 1.0, 0.0)

        corrected[i] = spectra[i] - baseline
        baselines[i] = baseline

    return corrected, baselines


def baseline_arpls(
    spectra: np.ndarray,
    lam: float = 1e4,
    ratio: float = 0.05,
    niter: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    arPLS (asymmetrically reweighted Penalized Least Squares)
    
    改进的AsLS，权重更新更稳定
    """
    L = spectra.shape[1]
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    H = lam * D.dot(D.T)

    corrected = np.zeros_like(spectra)
    baselines = np.zeros_like(spectra)

    for i in range(spectra.shape[0]):
        y = spectra[i]
        w = np.ones(L)
        for _ in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + H
            z = spsolve(Z, w * y)
            d = y - z
            d_neg = d[d < 0]
            m = np.mean(d_neg)
            s = np.std(d_neg)
            w_new = 1.0 / (1.0 + np.exp(2 * (d - (2 * s - m)) / s))
            if np.linalg.norm(w - w_new) / np.linalg.norm(w) < ratio:
                break
            w = w_new

        corrected[i] = y - z
        baselines[i] = z

    return corrected, baselines


# 便捷字典
BASELINE_METHODS = {
    "als": baseline_als,
    "airpls": baseline_airpls,
    "snip": baseline_snip,
    "modpoly": baseline_modpoly,
    "arpls": baseline_arpls,
}
