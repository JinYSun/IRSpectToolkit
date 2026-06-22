"""
数据质量评估 — 信噪比、重复性、完整性检查
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class QualityReport:
    """数据质量评估报告"""
    snr: float                    # 信噪比 (dB)
    repeatability_cv: float       # 重复性 (变异系数%)
    completeness: float           # 完整性 (0-1)
    score: float                  # 综合评分 (0-100)
    issues: list[str]             # 发现的问题
    suggestions: list[str]        # 改进建议


def snr_estimate(spectra: np.ndarray) -> float:
    """
    估算信噪比 (Signal-to-Noise Ratio)
    
    使用高频区域的标准差作为噪声估计
    """
    if spectra.ndim == 1:
        spectra = spectra.reshape(1, -1)

    mean_spectrum = spectra.mean(axis=0)
    signal_power = np.mean(mean_spectrum ** 2)

    # 噪声: 用相邻点差分的高频分量估计
    diff = np.diff(spectra, axis=1)
    noise_power = np.mean(diff ** 2) / 2

    if noise_power == 0:
        return 100.0  # 理想情况

    snr = 10 * np.log10(signal_power / noise_power)
    return round(snr, 1)


def repeatability_check(spectra: np.ndarray) -> float:
    """
    重复性检查 — 同一样本多次测量的一致性
    
    Returns:
        变异系数 (CV%) — 越小越好，<2%为优秀
    """
    if spectra.shape[0] < 2:
        return 0.0

    mean_spectrum = spectra.mean(axis=0)
    std_spectrum = spectra.std(axis=0)

    # 避免除零
    mean_safe = np.where(mean_spectrum == 0, 1e-10, mean_spectrum)
    cv = np.mean(std_spectrum / np.abs(mean_safe)) * 100

    return round(cv, 2)


def completeness_check(
    spectra: np.ndarray,
    wavelengths: np.ndarray | None = None,
    expected_range: tuple[float, float] | None = None,
) -> dict:
    """
    完整性检查
    
    检查内容:
    1. NaN/Inf值
    2. 全零光谱
    3. 波长范围覆盖
    """
    issues = []

    # NaN检查
    nan_count = np.isnan(spectra).sum()
    if nan_count > 0:
        issues.append(f"发现 {nan_count} 个NaN值")

    # Inf检查
    inf_count = np.isinf(spectra).sum()
    if inf_count > 0:
        issues.append(f"发现 {inf_count} 个Inf值")

    # 全零光谱
    zero_spectra = np.where(np.all(spectra == 0, axis=1))[0]
    if len(zero_spectra) > 0:
        issues.append(f"第 {zero_spectra.tolist()} 条光谱全为零")

    # 波长范围
    if wavelengths is not None and expected_range is not None:
        actual_range = (wavelengths.min(), wavelengths.max())
        if actual_range[0] > expected_range[0] or actual_range[1] < expected_range[1]:
            issues.append(
                f"波长范围 {actual_range} 不覆盖期望范围 {expected_range}"
            )

    # 完整性分数
    total_points = spectra.size
    bad_points = nan_count + inf_count + len(zero_spectra) * spectra.shape[1]
    completeness = 1.0 - bad_points / total_points if total_points > 0 else 0.0

    return {
        "completeness": round(completeness, 4),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "zero_spectra_count": len(zero_spectra),
        "issues": issues,
    }


def spectral_quality_score(
    spectra: np.ndarray,
    wavelengths: np.ndarray | None = None,
    expected_range: tuple[float, float] | None = None,
) -> QualityReport:
    """
    综合数据质量评估
    
    Returns:
        QualityReport 包含评分和建议
    """
    issues = []
    suggestions = []

    # SNR
    snr = snr_estimate(spectra)
    if snr < 20:
        issues.append(f"信噪比偏低: {snr} dB")
        suggestions.append("建议增加扫描次数或检查仪器状态")

    # 重复性
    cv = repeatability_check(spectra)
    if cv > 5:
        issues.append(f"重复性较差: CV={cv}%")
        suggestions.append("建议检查样品制备和仪器稳定性")
    elif cv > 2:
        issues.append(f"重复性一般: CV={cv}%")

    # 完整性
    comp_result = completeness_check(spectra, wavelengths, expected_range)
    completeness = comp_result["completeness"]
    issues.extend(comp_result["issues"])

    if comp_result["nan_count"] > 0:
        suggestions.append("建议用插值或删除处理NaN值")

    # 综合评分 (0-100)
    snr_score = min(snr / 40 * 100, 100)  # 40dB为满分
    cv_score = max(100 - cv * 10, 0)       # CV<10%为满分
    comp_score = completeness * 100

    overall = snr_score * 0.3 + cv_score * 0.3 + comp_score * 0.4

    return QualityReport(
        snr=snr,
        repeatability_cv=cv,
        completeness=completeness,
        score=round(overall, 1),
        issues=issues,
        suggestions=suggestions,
    )
