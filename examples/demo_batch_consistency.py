"""
示例5: 批次一致性检查 — 化工原料质量控制
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")

np.random.seed(42)
n_wavelengths = 500
wavelengths = np.linspace(900, 2500, n_wavelengths)

# ============================================================
# 模拟5个批次的原料近红外光谱
# ============================================================
def make_batch(batch_id, n_spectra=10, drift=0.0):
    """模拟批次光谱 (batch_id影响轻微偏移)"""
    spectra = []
    for _ in range(n_spectra):
        spec = 0.5 * np.exp(-0.5 * ((wavelengths - 1200) / 50) ** 2)
        spec += 0.3 * np.exp(-0.5 * ((wavelengths - 1450) / 30) ** 2)
        spec += 0.4 * np.exp(-0.5 * ((wavelengths - 1940) / 40) ** 2)
        # 批次漂移
        spec += drift * np.sin(wavelengths / 200)
        # 随机噪声
        spec += np.random.normal(0, 0.01, n_wavelengths)
        spectra.append(spec)
    return np.array(spectra)

batch_1 = make_batch(1, 10, drift=0.0)
batch_2 = make_batch(2, 10, drift=0.01)
batch_3 = make_batch(3, 10, drift=0.02)
batch_4 = make_batch(4, 10, drift=0.15)  # 异常批次
batch_5 = make_batch(5, 10, drift=0.01)

spectra = np.vstack([batch_1, batch_2, batch_3, batch_4, batch_5])
batch_labels = np.array([1]*10 + [2]*10 + [3]*10 + [4]*10 + [5]*10)

print("=" * 60)
print("  批次一致性检查示例")
print("=" * 60)
print(f"数据: {spectra.shape[0]} 条光谱, 5个批次")

# ============================================================
# 批次一致性检查
# ============================================================
from irspectoolkit.skills.identification import batch_consistency

result = batch_consistency(spectra, batch_labels)
print(f"\n{result.report}")

# ============================================================
# 异常筛查
# ============================================================
from irspectoolkit.skills.screening import anomaly_screen

anomaly_result = anomaly_screen(spectra)
print(f"\n{anomaly_result.conclusion}")
print(anomaly_result.report)

# ============================================================
# 可视化
# ============================================================
from irspectoolkit.visualization.plots import plot_spectra
from irspectoolkit.analysis.dimension import reduce_pca

pca_result = reduce_pca(spectra, n_components=3)
from irspectoolkit.visualization.plots import plot_pca_scores

fig1 = plot_spectra(spectra, wavelengths, labels=[f"Batch_{b}" for b in batch_labels], title="All Batches")
fig1.savefig("08_batch_spectra.png", dpi=150)

fig2 = plot_pca_scores(pca_result["scores"], batch_labels, pca_result["explained_variance_ratio"], title="Batch PCA")
fig2.savefig("09_batch_pca.png", dpi=150)

print("\n图表已保存: 08_batch_spectra.png, 09_batch_pca.png")
