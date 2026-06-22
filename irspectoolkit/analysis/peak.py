"""峰分析 — 检测、归属、积分"""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import trapezoid
import json
from pathlib import Path


def detect_peaks(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    height: float | None = None,
    prominence: float | None = None,
    distance: int | None = None,
) -> list[dict]:
    """
    峰检测
    
    Returns:
        [{"position": float, "wavelength": float, "height": float, "width": float, "prominence": float}, ...]
    """
    if height is None:
        height = spectrum.mean() + 0.5 * spectrum.std()
    if prominence is None:
        prominence = 0.1 * spectrum.std()
    if distance is None:
        distance = max(len(spectrum) // 100, 1)

    peaks, properties = find_peaks(
        spectrum, height=height, prominence=prominence, distance=distance
    )

    # 计算峰宽
    widths_result = peak_widths(spectrum, peaks, rel_height=0.5)

    results = []
    for i, idx in enumerate(peaks):
        results.append({
            "index": int(idx),
            "wavelength": float(wavelengths[idx]),
            "height": float(spectrum[idx]),
            "width": float(widths_result[0][i]),
            "prominence": float(properties["prominences"][i]) if "prominences" in properties else 0.0,
        })

    return results


def assign_functional_groups(
    peaks: list[dict],
    knowledge_base: str | dict | None = None,
) -> list[dict]:
    """
    峰归属 — 根据波长位置匹配官能团
    
    Args:
        peaks: detect_peaks的输出
        knowledge_base: 官能团知识库路径或字典
    """
    # 默认官能团-峰位映射 (cm⁻¹)
    default_groups = {
        "O-H (醇)": {"range": (3200, 3550), "shape": "broad"},
        "O-H (羧酸)": {"range": (2500, 3300), "shape": "very_broad"},
        "N-H (胺)": {"range": (3300, 3500), "shape": "medium"},
        "C-H (烷烃)": {"range": (2850, 3000), "shape": "sharp"},
        "C-H (烯烃)": {"range": (3000, 3100), "shape": "sharp"},
        "C≡N (腈)": {"range": (2200, 2260), "shape": "sharp"},
        "C=O (酮)": {"range": (1705, 1725), "shape": "sharp"},
        "C=O (醛)": {"range": (1720, 1740), "shape": "sharp"},
        "C=O (酯)": {"range": (1735, 1750), "shape": "sharp"},
        "C=O (酰胺)": {"range": (1630, 1690), "shape": "sharp"},
        "C=C (烯)": {"range": (1620, 1680), "shape": "medium"},
        "C=C (芳环)": {"range": (1450, 1600), "shape": "medium"},
        "C-O (醚)": {"range": (1000, 1300), "shape": "strong"},
        "C-F": {"range": (1000, 1400), "shape": "strong"},
        "C-Cl": {"range": (550, 850), "shape": "strong"},
        "N-H (伯胺)": {"range": (1560, 1650), "shape": "medium"},
        "N-O (硝基)": {"range": (1300, 1370), "shape": "strong"},
        "CH₂ (弯曲)": {"range": (1450, 1470), "shape": "medium"},
        "CH₃ (弯曲)": {"range": (1370, 1390), "shape": "medium"},
    }

    if knowledge_base is not None:
        if isinstance(knowledge_base, str):
            with open(knowledge_base, "r") as f:
                groups = json.load(f)
        else:
            groups = knowledge_base
    else:
        groups = default_groups

    assignments = []
    for peak in peaks:
        wl = peak["wavelength"]
        matched = []
        for group_name, info in groups.items():
            r = info["range"] if isinstance(info, dict) else info
            if r[0] <= wl <= r[1]:
                matched.append(group_name)

        assignments.append({
            **peak,
            "assignments": matched if matched else ["unassigned"],
        })

    return assignments


def integrate_peak_area(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    w_min: float,
    w_max: float,
) -> dict:
    """
    峰面积积分 — 梯形法则
    """
    mask = (wavelengths >= w_min) & (wavelengths <= w_max)
    wl_region = wavelengths[mask]
    spec_region = spectrum[mask]

    if len(wl_region) < 2:
        return {"area": 0.0, "range": (w_min, w_max), "points": 0}

    # 基线校正 (端点连线)
    baseline = np.linspace(spec_region[0], spec_region[-1], len(spec_region))
    corrected = spec_region - baseline
    corrected = np.maximum(corrected, 0)  # 只积分正值

    area = trapezoid(corrected, wl_region)

    return {
        "area": round(float(area), 4),
        "range": (w_min, w_max),
        "points": len(wl_region),
        "max_intensity": round(float(corrected.max()), 4),
        "peak_wavelength": round(float(wl_region[corrected.argmax()]), 1),
    }
