"""
示例4: 知识驱动峰归属 — 自动识别官能团
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")

np.random.seed(42)
n_wavelengths = 1000
wavelengths = np.linspace(400, 4000, n_wavelengths)

# ============================================================
# 模拟含多种官能团的化合物光谱 (如乙酸乙酯)
# ============================================================
spectrum = np.random.normal(0, 0.005, n_wavelengths)
spectrum += 0.95 * np.exp(-0.5 * ((wavelengths - 1740) / 15) ** 2)  # C=O (酯)
spectrum += 0.6 * np.exp(-0.5 * ((wavelengths - 1240) / 25) ** 2)   # C-O (酯)
spectrum += 0.5 * np.exp(-0.5 * ((wavelengths - 2980) / 30) ** 2)   # C-H
spectrum += 0.4 * np.exp(-0.5 * ((wavelengths - 1375) / 10) ** 2)   # CH3
spectrum += 0.3 * np.exp(-0.5 * ((wavelengths - 1460) / 12) ** 2)   # CH2

print("=" * 60)
print("  知识驱动峰归属示例")
print("=" * 60)

# ============================================================
# 官能团扫描
# ============================================================
from irspectoolkit.skills.knowledge_driven import functional_group_scan
from irspectoolkit.intelligence.knowledge import SpectralKnowledgeBase

kb = SpectralKnowledgeBase()
result = functional_group_scan(spectrum, wavelengths, knowledge_base=kb)

print(f"\n{result.report}")

# ============================================================
# 自定义知识库
# ============================================================
print("\n--- 自定义知识库查询 ---")
print(f"查询 1740 cm⁻¹ 附近的官能团:")
matches = kb.query_by_wavelength(1740, tolerance=20)
for m in matches:
    print(f"  {m['group']}: {m['description']}")

print(f"\n查询关键词 '酮':")
results = kb.query_by_keyword("酮")
for r in results:
    print(f"  {r['group']}")

# ============================================================
# 可视化
# ============================================================
from irspectoolkit.visualization.plots import plot_peak_assignment

fig = plot_peak_assignment(spectrum, wavelengths, result.details["assignments"], title="Ethyl Acetate - Peak Assignment")
fig.savefig("07_peak_assignment.png", dpi=150)
print("\n图表已保存: 07_peak_assignment.png")
