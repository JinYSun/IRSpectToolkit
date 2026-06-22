"""
示例2: PLS浓度预测 — 定量分析全流程
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# 生成模拟浓度-光谱数据
# ============================================================
n_samples = 100
n_wavelengths = 400
wavelengths = np.linspace(900, 1700, n_wavelengths)  # NIR范围

# 真实浓度 (0-100%)
concentrations = np.random.uniform(0, 100, n_samples)

# 光谱 = 基线 + 浓度相关峰 + 噪声
spectra = np.zeros((n_samples, n_wavelengths))
for i, c in enumerate(concentrations):
    base = 0.3 * np.exp(-0.5 * ((wavelengths - 1200) / 80) ** 2)
    peak1 = (c / 100) * 0.8 * np.exp(-0.5 * ((wavelengths - 1400) / 20) ** 2)
    peak2 = (c / 100) * 0.5 * np.exp(-0.5 * ((wavelengths - 1550) / 30) ** 2)
    noise = np.random.normal(0, 0.01, n_wavelengths)
    spectra[i] = base + peak1 + peak2 + noise

# 划分训练/测试
train_idx = np.arange(80)
test_idx = np.arange(80, 100)
X_train, X_test = spectra[train_idx], spectra[test_idx]
y_train, y_test = concentrations[train_idx], concentrations[test_idx]

print("=" * 60)
print("  PLS浓度预测示例")
print("=" * 60)

# ============================================================
# 预处理 + PLS回归
# ============================================================
from irspectoolkit.preprocessing.transform import snv, savgol_derivative
from irspectoolkit.skills.quantification import concentration_predict

result = concentration_predict(X_train, y_train, X_test=X_test, y_test=y_test)
print(f"\n{result.report}")

# ============================================================
# 可视化
# ============================================================
from irspectoolkit.visualization.plots import plot_regression

X_train_processed = savgol_derivative(snv(X_train))
model = result.raw_data["model"]
y_pred = model.predict(X_train_processed).flatten()

fig = plot_regression(y_train, y_pred, title="PLS Concentration Prediction")
fig.savefig("05_pls_regression.png", dpi=150)
print("\n图表已保存: 05_pls_regression.png")
