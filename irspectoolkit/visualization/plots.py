"""多类型光谱可视化图表"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_spectra(spectra, wavelengths, labels=None, title="Spectra", n_max=20):
    """光谱叠加图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    n = min(spectra.shape[0], n_max)
    for i in range(n):
        label = labels[i] if labels is not None else None
        ax.plot(wavelengths, spectra[i], alpha=0.6, label=label)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    if labels is not None:
        ax.legend(fontsize=8, ncol=2)
    ax.invert_xaxis()
    plt.tight_layout()
    return fig


def plot_pca_scores(scores, labels=None, explained_var=None, title="PCA Scores"):
    """PCA得分图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    classes = np.unique(labels) if labels is not None else [None]
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for cls, color in zip(classes, colors):
        if cls is not None:
            mask = labels == cls
            ax.scatter(scores[mask, 0], scores[mask, 1], c=[color], label=str(cls), alpha=0.7, s=50)
        else:
            ax.scatter(scores[:, 0], scores[:, 1], alpha=0.7, s=50)

    xlabel = f"PC1 ({explained_var[0]*100:.1f}%)" if explained_var is not None else "PC1"
    ylabel = f"PC2 ({explained_var[1]*100:.1f}%)" if explained_var is not None else "PC2"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if labels is not None:
        ax.legend()
    plt.tight_layout()
    return fig


def plot_pca_loadings(loadings, wavelengths, n_components=3, title="PCA Loadings"):
    """PCA载荷图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(min(n_components, loadings.shape[0])):
        ax.plot(wavelengths, loadings[i], label=f"PC{i+1}")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Loading")
    ax.set_title(title)
    ax.legend()
    ax.invert_xaxis()
    ax.axhline(y=0, color="k", linewidth=0.5)
    plt.tight_layout()
    return fig


def plot_baseline_comparison(original, corrected, baseline, wavelengths, title="Baseline Correction"):
    """基线校正前后对比图"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(wavelengths, original, label="Original", alpha=0.8)
    axes[0].plot(wavelengths, baseline, label="Baseline", linestyle="--", color="red")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title(title)
    axes[0].legend()

    axes[1].plot(wavelengths, corrected, label="Corrected", color="green")
    axes[1].set_xlabel("Wavenumber (cm⁻¹)")
    axes[1].set_ylabel("Intensity")
    axes[1].legend()

    for ax in axes:
        ax.invert_xaxis()
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """混淆矩阵"""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_regression(y_true, y_pred, title="Regression"):
    """回归预测 vs 实际值散点图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Ideal")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.text(0.05, 0.95, f"R²={r2:.4f}\nRMSE={rmse:.4f}", transform=ax.transAxes,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat"))
    plt.tight_layout()
    return fig


def plot_peak_assignment(spectrum, wavelengths, peaks, title="Peak Assignment"):
    """峰标注图"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wavelengths, spectrum, "b-", linewidth=1)
    for peak in peaks:
        wl = peak["wavelength"]
        idx = peak.get("index", 0)
        h = peak.get("height", spectrum[idx])
        ax.plot(wl, h, "rv", markersize=8)
        label = ", ".join(peak.get("assignments", peak.get("group", ["?"]))) if "assignments" in peak or "group" in peak else ""
        if label:
            ax.annotate(label, (wl, h), textcoords="offset points", xytext=(0, 10),
                       ha="center", fontsize=7, rotation=45)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.invert_xaxis()
    plt.tight_layout()
    return fig


def plot_library_match(query, matches, library, wavelengths=None, top_n=3, title="Library Search"):
    """光谱库搜索匹配结果图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    if wavelengths is None:
        wavelengths = np.arange(len(query))

    ax.plot(wavelengths, query, "k-", linewidth=2, label="Query", zorder=10)
    colors = ["r", "g", "b", "orange", "purple"]
    for i, match in enumerate(matches[:top_n]):
        idx = match["index"]
        score = match["score"]
        name = match.get("name", f"#{idx}")
        ax.plot(wavelengths, library[idx], color=colors[i % len(colors)],
                linestyle="--", alpha=0.7, label=f"{name} (score={score:.3f})")

    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.invert_xaxis()
    plt.tight_layout()
    return fig


def plot_quality_report(report, title="Data Quality Report"):
    """数据质量评估可视化"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # SNR
    axes[0].bar(["SNR"], [report.snr], color="steelblue")
    axes[0].axhline(y=30, color="r", linestyle="--", label="Good (30dB)")
    axes[0].set_ylabel("dB")
    axes[0].set_title("Signal-to-Noise Ratio")
    axes[0].legend()

    # CV
    axes[1].bar(["CV%"], [report.repeatability_cv], color="coral")
    axes[1].axhline(y=2, color="g", linestyle="--", label="Excellent (2%)")
    axes[1].set_ylabel("%")
    axes[1].set_title("Repeatability")
    axes[1].legend()

    # Overall score
    score = report.score
    color = "green" if score > 80 else "orange" if score > 50 else "red"
    axes[2].bar(["Score"], [score], color=color)
    axes[2].set_ylim(0, 100)
    axes[2].set_title(f"Overall: {score:.0f}/100")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig
