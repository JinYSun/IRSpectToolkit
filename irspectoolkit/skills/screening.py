"""异常筛查Skills"""
from __future__ import annotations
import numpy as np
from .base import SkillResult
from irspectoolkit.preprocessing.transform import snv
from irspectoolkit.preprocessing.outlier import (
    detect_outliers_mahalanobis, detect_outliers_lof, detect_outliers_pca_q
)


def anomaly_screen(
    spectra: np.ndarray,
    methods: list[str] = ["mahalanobis", "lof", "pca_q"],
    preprocess: bool = True,
) -> SkillResult:
    """
    异常筛查: 多方法投票
    """
    if preprocess:
        spectra = snv(spectra)

    all_outliers = {}
    vote_count = np.zeros(spectra.shape[0])

    if "mahalanobis" in methods:
        idx, dists = detect_outliers_mahalanobis(spectra)
        all_outliers["mahalanobis"] = idx.tolist()
        vote_count[idx] += 1

    if "lof" in methods:
        idx, scores = detect_outliers_lof(spectra)
        all_outliers["lof"] = idx.tolist()
        vote_count[idx] += 1

    if "pca_q" in methods:
        idx, q_stats = detect_outliers_pca_q(spectra)
        all_outliers["pca_q"] = idx.tolist()
        vote_count[idx] += 1

    # 投票: 多数方法认为异常才标记
    consensus_threshold = len(methods) / 2
    consensus_outliers = np.where(vote_count > consensus_threshold)[0]

    conclusion = f"共 {spectra.shape[0]} 条光谱, 发现 {len(consensus_outliers)} 条异常"
    if len(consensus_outliers) > 0:
        conclusion += f": 样本 {consensus_outliers.tolist()}"

    report_lines = [conclusion, "\n各方法检测结果:"]
    for method, idx in all_outliers.items():
        report_lines.append(f"  {method}: {len(idx)} outliers")

    return SkillResult(
        conclusion=conclusion,
        confidence=1.0 - len(consensus_outliers) / spectra.shape[0],
        details={
            "outlier_indices": consensus_outliers.tolist(),
            "vote_count": vote_count.tolist(),
            "method_results": all_outliers,
        },
        report="\n".join(report_lines),
        raw_data=all_outliers,
    )
