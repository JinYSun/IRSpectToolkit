"""
示例6: 参数自动寻优 — Optuna贝叶斯优化最佳pipeline
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")

np.random.seed(42)

# ============================================================
# 生成3类光谱数据
# ============================================================
n_per_class = 40
n_wavelengths = 300
wavelengths = np.linspace(800, 2500, n_wavelengths)

def make_spectrum(cls, noise=0.02):
    spec = np.random.normal(0, noise, n_wavelengths)
    if cls == 0:
        spec += 0.8 * np.exp(-0.5 * ((wavelengths - 1200) / 30) ** 2)
        spec += 0.5 * np.exp(-0.5 * ((wavelengths - 1700) / 25) ** 2)
    elif cls == 1:
        spec += 0.7 * np.exp(-0.5 * ((wavelengths - 1100) / 35) ** 2)
        spec += 0.6 * np.exp(-0.5 * ((wavelengths - 1800) / 20) ** 2)
    else:
        spec += 0.6 * np.exp(-0.5 * ((wavelengths - 1300) / 40) ** 2)
        spec += 0.8 * np.exp(-0.5 * ((wavelengths - 1600) / 30) ** 2)
    return spec

spectra = []
labels = []
for cls in range(3):
    for _ in range(n_per_class):
        spectra.append(make_spectrum(cls))
        labels.append(cls)
spectra = np.array(spectra)
labels = np.array(labels)

print("=" * 60)
print("  参数自动寻优示例 (Optuna)")
print("=" * 60)

# ============================================================
# 自动优化
# ============================================================
try:
    from irspectoolkit.intelligence.optimizer import SpectralOptimizer

    optimizer = SpectralOptimizer(spectra, labels, task="classification")
    result = optimizer.optimize_classification(n_trials=30, cv_folds=5)

    print(f"\n最佳得分: {result['best_score']:.4f}")
    print(f"最佳参数: {result['best_params']}")

    # 手动对比: 不优化直接用默认参数
    from irspectoolkit.preprocessing.transform import snv, savgol_derivative
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.svm import SVC

    spectra_default = savgol_derivative(snv(spectra))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    default_score = cross_val_score(SVC(random_state=42), spectra_default, labels, cv=cv).mean()

    print(f"\n对比:")
    print(f"  默认参数: {default_score:.4f}")
    print(f"  优化后:   {result['best_score']:.4f}")
    print(f"  提升:     {(result['best_score'] - default_score)*100:.1f}%")

except ImportError:
    print("\n需要安装 optuna: pip install optuna")
    print("跳过寻优示例")
