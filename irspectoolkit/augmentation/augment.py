"""
光谱数据增强 — 解决小样本问题
支持: 加噪、波长抖动、Mixup、光谱混合
"""
from __future__ import annotations
import numpy as np


def add_noise(
    spectra: np.ndarray,
    noise_level: float = 0.01,
    n_augmented: int = 1,
) -> np.ndarray:
    """
    高斯噪声注入
    
    Args:
        noise_level: 噪声标准差占信号标准差的比例
        n_augmented: 每条光谱生成多少条增强数据
    """
    augmented = []
    for _ in range(n_augmented):
        noise = np.random.normal(0, noise_level * spectra.std(), spectra.shape)
        augmented.append(spectra + noise)
    return np.vstack(augmented)


def wavelength_jitter(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    max_shift: int = 3,
    n_augmented: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    波长轴抖动 — 模拟仪器校准差异
    
    通过插值实现波长方向的随机偏移
    """
    from scipy.interpolate import interp1d

    augmented_spectra = []
    n_points = len(wavelengths)

    for _ in range(n_augmented * spectra.shape[0]):
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            augmented_spectra.append(spectra[np.random.randint(spectra.shape[0])])
            continue

        # 选择一条随机光谱
        idx = np.random.randint(spectra.shape[0])
        spec = spectra[idx]

        # 通过插值模拟波长偏移
        new_wl = wavelengths + shift * (wavelengths[1] - wavelengths[0])
        f = interp1d(wavelengths, spec, kind="linear", fill_value="extrapolate")
        augmented_spectra.append(f(new_wl))

    return np.array(augmented_spectra), wavelengths


def spectral_mixup(
    spectra: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.2,
    n_augmented: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    光谱Mixup — 线性插值生成新样本
    
    x_new = λ * x_i + (1-λ) * x_j
    y_new = λ * y_i + (1-λ) * y_j
    """
    n = spectra.shape[0]
    if n_augmented is None:
        n_augmented = n

    augmented_spectra = []
    augmented_labels = []

    for _ in range(n_augmented):
        i, j = np.random.choice(n, 2, replace=False)
        lam = np.random.beta(alpha, alpha)
        new_spec = lam * spectra[i] + (1 - lam) * spectra[j]

        # 混合标签 (对one-hot有效)
        if labels.ndim > 1:
            new_label = lam * labels[i] + (1 - lam) * labels[j]
        else:
            new_label = labels[i] if lam > 0.5 else labels[j]

        augmented_spectra.append(new_spec)
        augmented_labels.append(new_label)

    return np.array(augmented_spectra), np.array(augmented_labels)


def generate_mixture(
    pure_spectra: np.ndarray,
    n_mixtures: int = 100,
    n_components_range: tuple[int, int] = (2, 4),
) -> tuple[np.ndarray, np.ndarray]:
    """
    从纯组分光谱生成混合物光谱
    
    Args:
        pure_spectra: (n_pure, n_wavelengths) 纯组分光谱
        n_mixtures: 生成混合物数量
        n_components_range: 每个混合物包含的组分数
    
    Returns:
        (mixtures, concentrations) — 混合光谱和对应浓度
    """
    n_pure = pure_spectra.shape[0]
    mixtures = []
    concentrations = []

    for _ in range(n_mixtures):
        n_comp = np.random.randint(n_components_range[0], n_components_range[1] + 1)
        n_comp = min(n_comp, n_pure)

        # 随机选择组分
        comp_indices = np.random.choice(n_pure, n_comp, replace=False)

        # 随机浓度 (Dirichlet分布保证和为1)
        conc = np.random.dirichlet(np.ones(n_comp))

        # 混合
        mixture = np.zeros(pure_spectra.shape[1])
        full_conc = np.zeros(n_pure)
        for idx, c in zip(comp_indices, conc):
            mixture += c * pure_spectra[idx]
            full_conc[idx] = c

        mixtures.append(mixture)
        concentrations.append(full_conc)

    return np.array(mixtures), np.array(concentrations)


def augment_dataset(
    spectra: np.ndarray,
    labels: np.ndarray | None = None,
    methods: list[str] = ["noise", "mixup"],
    noise_level: float = 0.01,
    mixup_alpha: float = 0.2,
    multiplier: int = 3,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    一键数据增强
    
    Args:
        methods: 使用的增强方法列表
        multiplier: 增强倍数 (最终数据量 = 原始 * multiplier)
    
    Returns:
        (augmented_spectra, augmented_labels)
    """
    all_spectra = [spectra]
    all_labels = [labels] if labels is not None else [None]

    n_per_method = max(1, (multiplier - 1) * spectra.shape[0] // len(methods))

    for method in methods:
        if method == "noise":
            aug = add_noise(spectra, noise_level, n_augmented=1)
            all_spectra.append(aug)
            if labels is not None:
                all_labels.append(np.tile(labels, 1))
            else:
                all_labels.append(None)

        elif method == "mixup" and labels is not None:
            aug_s, aug_l = spectral_mixup(spectra, labels, alpha=mixup_alpha, n_augmented=n_per_method)
            all_spectra.append(aug_s)
            all_labels.append(aug_l)

    combined_spectra = np.vstack(all_spectra)
    if labels is not None:
        combined_labels = np.concatenate([l for l in all_labels if l is not None])
    else:
        combined_labels = None

    return combined_spectra, combined_labels
