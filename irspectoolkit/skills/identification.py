"""定性鉴别Skills"""
from __future__ import annotations
import numpy as np
from .base import SkillResult
from irspectoolkit.preprocessing.transform import snv, savgol_derivative
from irspectoolkit.analysis.library import SpectralLibrary, hqi_search, cosine_search
from irspectoolkit.analysis.dimension import reduce_pca
from irspectoolkit.preprocessing.outlier import detect_outliers_mahalanobis


def rapid_identify(
    spectrum: np.ndarray,
    library: SpectralLibrary,
    wavelengths: np.ndarray | None = None,
    preprocess: bool = True,
    top_k: int = 5,
) -> SkillResult:
    """
    快速鉴别: "这是什么？"
    
    自动预处理 → 光谱库搜索 → 返回Top-N匹配
    """
    spec = spectrum.reshape(1, -1) if spectrum.ndim == 1 else spectrum

    if preprocess:
        spec = snv(spec)
        spec = savgol_derivative(spec, window_length=11, polyorder=2, deriv=1)

    query = spec.flatten()
    results = library.search(query, method="hqi", top_k=top_k)

    best = results[0] if results else {"name": "Unknown", "score": 0}
    conclusion = f"最匹配: {best['name']} (HQI={best['score']:.3f})"

    report_lines = [f"光谱库搜索结果 (Top-{top_k}):"]
    for i, r in enumerate(results):
        report_lines.append(f"  {i+1}. {r['name']} — HQI={r['score']:.3f}")

    return SkillResult(
        conclusion=conclusion,
        confidence=best["score"],
        details={"top_k_results": results},
        report="\n".join(report_lines),
        raw_data={"search_results": results},
    )


def authenticity_check(
    spectrum: np.ndarray,
    reference_spectra: np.ndarray,
    reference_name: str = "Standard",
    threshold: float = 3.0,
    preprocess: bool = True,
) -> SkillResult:
    """
    真伪鉴定: 和标准谱对比，判断是否合格
    """
    specs = np.vstack([spectrum.reshape(1, -1), reference_spectra])

    if preprocess:
        specs = snv(specs)
        specs = savgol_derivative(specs)

    # PCA + 马氏距离
    result = reduce_pca(specs, n_components=min(3, specs.shape[0] - 1))
    scores = result["scores"]

    # 测试样本到参考集的马氏距离
    ref_scores = scores[1:]
    test_score = scores[0]

    try:
        from sklearn.covariance import MinCovDet
        robust_cov = MinCovDet().fit(ref_scores)
        dist = robust_cov.mahalanobis(test_score.reshape(1, -1))[0]
    except Exception:
        mean = ref_scores.mean(axis=0)
        cov = np.cov(ref_scores.T)
        diff = test_score - mean
        dist = np.sqrt(diff @ np.linalg.pinv(cov) @ diff)

    passed = dist < threshold
    conclusion = f"{'✅ 合格' if passed else '❌ 不合格'} — 与{reference_name}的马氏距离={dist:.2f} (阈值={threshold})"

    return SkillResult(
        conclusion=conclusion,
        confidence=max(0, 1 - dist / (2 * threshold)),
        details={"mahalanobis_distance": float(dist), "threshold": threshold, "passed": passed},
        report=conclusion,
        raw_data={"pca_scores": scores, "distance": dist},
    )


def batch_consistency(
    spectra: np.ndarray,
    batch_labels: np.ndarray,
    preprocess: bool = True,
) -> SkillResult:
    """
    批次一致性检查: 这几批料一样吗？
    """
    if preprocess:
        spectra = snv(spectra)

    result = reduce_pca(spectra, n_components=3)
    scores = result["scores"]
    explained = result["explained_variance_ratio"]

    # 批间差异分析
    batches = np.unique(batch_labels)
    batch_means = {}
    for b in batches:
        mask = batch_labels == b
        batch_means[b] = scores[mask].mean(axis=0)

    # 计算批次间距离矩阵
    from scipy.spatial.distance import pdist, squareform
    mean_scores = np.array([batch_means[b] for b in batches])
    dist_matrix = squareform(pdist(mean_scores))

    max_dist = dist_matrix.max()
    consistent = max_dist < 5.0  # 经验阈值

    conclusion = f"批次{'一致' if consistent else '存在差异'} (最大批间距离={max_dist:.2f})"

    report_lines = [conclusion, "\n批间距离矩阵:"]
    for i, b1 in enumerate(batches):
        for j, b2 in enumerate(batches):
            if i < j:
                report_lines.append(f"  {b1} vs {b2}: {dist_matrix[i,j]:.2f}")

    return SkillResult(
        conclusion=conclusion,
        confidence=1.0 if consistent else 0.5,
        details={"distance_matrix": dist_matrix.tolist(), "batches": batches.tolist()},
        report="\n".join(report_lines),
        raw_data={"pca_scores": scores, "batch_means": batch_means},
    )
