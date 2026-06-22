"""域适应与迁移学习 — 跨仪器光谱对齐"""
from __future__ import annotations
import numpy as np


def spectral_align(
    source: np.ndarray,
    target: np.ndarray,
    method: str = "mean_centering",
) -> np.ndarray:
    """
    光谱对齐 — 将source域的光谱对齐到target域
    
    方法:
    - mean_centering: 均值中心化
    - zscore: Z分数标准化
    """
    if method == "mean_centering":
        source_mean = source.mean(axis=0)
        target_mean = target.mean(axis=0)
        return source - source_mean + target_mean
    elif method == "zscore":
        source_mean = source.mean(axis=0)
        source_std = source.std(axis=0)
        target_mean = target.mean(axis=0)
        target_std = target.std(axis=0)
        source_std[source_std == 0] = 1.0
        return (source - source_mean) / source_std * target_std + target_mean
    else:
        raise ValueError(f"不支持的对齐方法: {method}")


def tca_transform(
    source: np.ndarray,
    target: np.ndarray,
    n_components: int = 10,
    kernel: str = "rbf",
    mu: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    TCA (Transfer Component Analysis)
    
    在再生核希尔伯特空间中最小化域间分布差异
    
    简化实现: PCA + 均值对齐
    完整TCA需要求解广义特征值问题
    """
    from sklearn.decomposition import PCA

    combined = np.vstack([source, target])
    pca = PCA(n_components=n_components)
    combined_transformed = pca.fit_transform(combined)

    source_t = combined_transformed[:len(source)]
    target_t = combined_transformed[len(source):]

    # 均值对齐
    source_t = spectral_align(source_t, target_t, "mean_centering")

    return source_t, target_t
