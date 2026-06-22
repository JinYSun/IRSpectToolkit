"""
示例3: 药品真伪鉴别 — 真实场景模拟
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")

np.random.seed(42)
n_wavelengths = 500
wavelengths = np.linspace(400, 4000, n_wavelengths)

# ============================================================
# 模拟药品标准光谱和待测光谱
# ============================================================
def make_drug_spectrum(variant="standard", noise=0.01):
    """模拟药品红外光谱"""
    spec = np.random.normal(0, noise, n_wavelengths)
    # 主特征峰: 1650 (酰胺I), 1540 (酰胺II), 3300 (N-H)
    spec += 0.9 * np.exp(-0.5 * ((wavelengths - 1650) / 20) ** 2)
    spec += 0.6 * np.exp(-0.5 * ((wavelengths - 1540) / 25) ** 2)
    spec += 0.5 * np.exp(-0.5 * ((wavelengths - 3300) / 40) ** 2)

    if variant == "counterfeit":
        # 假药: 特征峰偏移 + 额外峰
        spec += 0.3 * np.exp(-0.5 * ((wavelengths - 1720) / 15) ** 2)
        spec *= 0.85  # 整体强度偏低
    elif variant == "expired":
        # 过期: 基线漂移
        spec += np.linspace(0, 0.15, n_wavelengths)

    return spec

# 标准谱 (30条)
standard_spectra = np.array([make_drug_spectrum("standard", 0.015) for _ in range(30)])

# 待测样品
test_genuine = make_drug_spectrum("standard", noise=0.02)
test_fake = make_drug_spectrum("counterfeit", noise=0.02)
test_expired = make_drug_spectrum("expired", noise=0.02)

print("=" * 60)
print("  药品真伪鉴别示例")
print("=" * 60)

# ============================================================
# 真伪鉴定
# ============================================================
from irspectoolkit.skills.identification import authenticity_check
from irspectoolkit.analysis.peak import detect_peaks, assign_functional_groups

print("\n--- 合格药品 ---")
r1 = authenticity_check(test_genuine, standard_spectra, "药品标准品")
print(r1.conclusion)

print("\n--- 假药 ---")
r2 = authenticity_check(test_fake, standard_spectra, "药品标准品")
print(r2.conclusion)

print("\n--- 过期药品 ---")
r3 = authenticity_check(test_expired, standard_spectra, "药品标准品")
print(r3.conclusion)

# ============================================================
# 峰归属分析
# ============================================================
from irspectoolkit.skills.knowledge_driven import functional_group_scan

print("\n--- 标准品官能团扫描 ---")
scan_result = functional_group_scan(make_drug_spectrum("standard", 0.005), wavelengths)
print(scan_result.report)

# ============================================================
# 可视化
# ============================================================
from irspectoolkit.visualization.plots import plot_spectra, plot_baseline_comparison

fig = plot_spectra(
    np.array([test_genuine, test_fake, test_expired]),
    wavelengths,
    labels=["Genuine", "Counterfeit", "Expired"],
    title="Drug Authenticity Comparison"
)
fig.savefig("06_drug_comparison.png", dpi=150)
print("\n图表已保存: 06_drug_comparison.png")
