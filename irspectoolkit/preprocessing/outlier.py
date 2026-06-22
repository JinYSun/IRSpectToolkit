"""
异常值检测 — 马氏距离, LOF, PCA-Q统计量, 孤立森林
"""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.covariance import MinCovDet


def detect_outliers_mahalanobis(
    spectra: np.ndarray,
    threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    马氏距离异常检测
    
    Returns:
        (outlier_indices, distances)
    """
    try:
        robust_cov = MinCovDet().fit(spectra)
        distances = robust_cov.mahalanobis(spectra)
    except Exception:
        # 回退到普通协方差
        mean = spectra.mean(axis=0)
        cov = np.cov(spectra.T)
        cov_inv = np.linalg.pinv(cov)
        diff = spectra - mean
        distances = np.array([d @ cov_inv @ d for d in diff])

    outlier_mask = distances > threshold ** 2
    return np.where(outlier_mask)[0], distances


def detect_outliers_lof(
    spectra: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Local Outlier Factor 异常检测
    
    Returns:
        (outlier_indices, scores)
    """
    n_neighbors = min(n_neighbors, spectra.shape[0] - 1)
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
    )
    labels = lof.fit_predict(spectra)
    scores = -lof.negative_outlier_factor_

    outlier_mask = labels == -1
    return np.where(outlier_mask)[0], scores


def detect_outliers_pca_q(
    spectra: np.ndarray,
    n_components: int = 5,
    threshold: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    PCA Q统计量 (SPE - Squared Prediction Error)
    
    超出主成分子空间的残差作为异常指标
    """
    n_components = min(n_components, spectra.shape[0] - 1, spectra.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(spectra)
    reconstructed = pca.inverse_transform(scores)
    residuals = spectra - reconstructed
    q_stats = np.sum(residuals ** 2, axis=1)

    # 基于卡方分布的阈值
    from scipy import stats
    q_threshold = np.percentile(q_stats, threshold * 100)

    outlier_mask = q_stats > q_threshold
    return np.where(outlier_mask)[0], q_stats


def detect_outliers_isolation_forest(
    spectra: np.ndarray,
    contamination: float = 0.1,
    n_estimators: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    孤立森林异常检测
    """
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=42,
    )
    labels = iso.fit_predict(spectra)
    scores = -iso.score_samples(spectra)

    outlier_mask = labels == -1
    return np.where(outlier_mask)[0], scores
