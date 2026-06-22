"""
示例1: 快速入门 — 合成光谱 + 预处理 + PCA + 分类
可在Spyder中直接运行
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互环境
import matplotlib.pyplot as plt

# ============================================================
# 生成模拟数据 — 3类物质的红外光谱
# ============================================================
np.random.seed(42)
n_samples_per_class = 30
n_wavelengths = 500
wavelengths = np.linspace(400, 4000, n_wavelengths)  # 400-4000 cm⁻¹

def make_spectrum(class_id, noise=0.02):
    """生成模拟光谱"""
    spec = np.random.normal(0, noise, n_wavelengths)
    if class_id == 0:  # 物质A: 强峰在1700
        spec += 0.8 * np.exp(-0.5 * ((wavelengths - 1700) / 30) ** 2)
        spec += 0.5 * np.exp(-0.5 * ((wavelengths - 2900) / 40) ** 2)
    elif class_id == 1:  # 物质B: 强峰在1600, 3300
        spec += 0.6 * np.exp(-0.5 * ((wavelengths - 1600) / 25) ** 2)
        spec += 0.7 * np.exp(-0.5 * ((wavelengths - 3300) / 50) ** 2)
    else:  # 物质C: 强峰在2200, 1200
        spec += 0.5 * np.exp(-0.5 * ((wavelengths - 2200) / 20) ** 2)
        spec += 0.6 * np.exp(-0.5 * ((wavelengths - 1200) / 35) ** 2)
    return spec

spectra = []
labels = []
for cls in range(3):
    for _ in range(n_samples_per_class):
        spectra.append(make_spectrum(cls))
        labels.append(cls)
spectra = np.array(spectra)
labels = np.array(labels)

print("=" * 60)
print("  IRSpecToolkit 快速入门示例")
print("=" * 60)
print(f"数据: {spectra.shape[0]} 条光谱 × {spectra.shape[1]} 个波长点")
print(f"类别: {np.unique(labels)}")

# ============================================================
# Step 1: 预处理
# ============================================================
from irspectoolkit.preprocessing.transform import snv, savgol_derivative
from irspectoolkit.preprocessing.baseline import baseline_als

# SNV
spectra_snv = snv(spectra)
print("\n[Step 1] SNV完成")

# 基线校正
spectra_bl, baselines = baseline_als(spectra, lam=1e6, p=0.01)
print("[Step 1] 基线校正 (AsLS) 完成")

# 一阶导数
spectra_deriv = savgol_derivative(spectra_snv, window_length=11, polyorder=2, deriv=1)
print("[Step 1] 一阶导数完成")

# ============================================================
# Step 2: 数据质量评估
# ============================================================
from irspectoolkit.quality.assessment import spectral_quality_score

quality = spectral_quality_score(spectra, wavelengths)
print(f"\n[Step 2] 数据质量评估:")
print(f"  SNR: {quality.snr} dB")
print(f"  重复性CV: {quality.repeatability_cv}%")
print(f"  综合评分: {quality.score}/100")

# ============================================================
# Step 3: 异常值检测
# ============================================================
from irspectoolkit.preprocessing.outlier import detect_outliers_lof

outlier_idx, scores = detect_outliers_lof(spectra_snv)
print(f"\n[Step 3] LOF异常检测: 发现 {len(outlier_idx)} 个异常样本")

# ============================================================
# Step 4: PCA降维
# ============================================================
from irspectoolkit.analysis.dimension import reduce_pca

pca_result = reduce_pca(spectra_deriv, n_components=3)
print(f"\n[Step 4] PCA降维:")
print(f"  累计方差解释率: {pca_result['cumulative_variance']}")

# ============================================================
# Step 5: 分类
# ============================================================
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

svm_scores = cross_val_score(SVC(kernel="rbf", C=10, random_state=42), spectra_deriv, labels, cv=cv)
knn_scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), spectra_deriv, labels, cv=cv)

print(f"\n[Step 5] 分类 (5-fold CV):")
print(f"  SVM准确率: {svm_scores.mean():.3f} ± {svm_scores.std():.3f}")
print(f"  KNN准确率: {knn_scores.mean():.3f} ± {knn_scores.std():.3f}")

# ============================================================
# Step 6: 峰归属
# ============================================================
from irspectoolkit.analysis.peak import detect_peaks, assign_functional_groups

mean_spectrum = spectra.mean(axis=0)
peaks = detect_peaks(mean_spectrum, wavelengths)
assignments = assign_functional_groups(peaks)

print(f"\n[Step 6] 峰归属:")
for a in assignments[:5]:
    groups = ", ".join(a["assignments"])
    print(f"  {a['wavelength']:.0f} cm⁻¹ → {groups}")

# ============================================================
# Step 7: 光谱库搜索
# ============================================================
from irspectoolkit.analysis.library import SpectralLibrary, hqi_search

library = SpectralLibrary()
for cls in range(3):
    library.add(f"Reference_Class{cls}", make_spectrum(cls, noise=0.01))

query = make_spectrum(0, noise=0.03)
results = library.search(query, method="hqi", top_k=3)
print(f"\n[Step 7] 光谱库搜索:")
for r in results:
    print(f"  {r['name']}: HQI={r['score']:.3f}")

# ============================================================
# Step 8: Skill调用
# ============================================================
from irspectoolkit.skills import rapid_identify, anomaly_screen

result = rapid_identify(make_spectrum(1, noise=0.02), library)
print(f"\n[Step 8] 快速鉴别Skill:")
print(f"  结论: {result.conclusion}")
print(f"  置信度: {result.confidence:.3f}")

# ============================================================
# Step 9: 可视化
# ============================================================
from irspectoolkit.visualization.plots import (
    plot_spectra, plot_pca_scores, plot_baseline_comparison, plot_peak_assignment
)

fig1 = plot_spectra(spectra, wavelengths, labels=labels, title="Raw Spectra")
fig1.savefig("01_raw_spectra.png", dpi=150)

fig2 = plot_pca_scores(pca_result["scores"], labels, pca_result["explained_variance_ratio"], title="PCA Scores")
fig2.savefig("02_pca_scores.png", dpi=150)

fig3 = plot_baseline_comparison(spectra[0], spectra_bl[0], baselines[0], wavelengths, title="Baseline Correction")
fig3.savefig("03_baseline.png", dpi=150)

fig4 = plot_peak_assignment(mean_spectrum, wavelengths, assignments, title="Peak Assignment")
fig4.savefig("04_peaks.png", dpi=150)

print(f"\n[Step 9] 图表已保存: 01-04_*.png")
print("\n" + "=" * 60)
print("  快速入门示例完成!")
print("=" * 60)
