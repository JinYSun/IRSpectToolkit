"""定量分析Skills"""
from __future__ import annotations
import numpy as np
from .base import SkillResult
from irspectoolkit.preprocessing.transform import snv, savgol_derivative
from irspectoolkit.analysis.regression import pls_regression


def concentration_predict(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    preprocess: bool = True,
    n_components: int = 10,
) -> SkillResult:
    """
    浓度预测: 用PLS回归预测含量
    """
    if preprocess:
        X = snv(X)
        X = savgol_derivative(X, window_length=11, polyorder=2, deriv=1)

    result = pls_regression(X, y, n_components=n_components)
    model = result["model"]

    report_lines = [
        f"PLS回归结果:",
        f"  最佳主成分数: {result['best_n_components']}",
        f"  RMSECV: {result['rmsecv']:.4f}",
        f"  R²(train): {result['r2_train']:.4f}",
    ]

    predictions = result["predictions_train"].tolist()

    if X_test is not None:
        if preprocess:
            X_test_processed = snv(X_test.reshape(1, -1) if X_test.ndim == 1 else X_test)
            X_test_processed = savgol_derivative(X_test_processed)
        else:
            X_test_processed = X_test
        y_pred = model.predict(X_test_processed.reshape(1, -1) if X_test_processed.ndim == 1 else X_test_processed).flatten()
        report_lines.append(f"  预测值: {y_pred[0]:.4f}")
        predictions = y_pred.tolist()

    return SkillResult(
        conclusion=f"PLS模型: R²={result['r2_train']:.4f}, RMSECV={result['rmsecv']:.4f}",
        confidence=result["r2_train"],
        details={
            "best_n_components": result["best_n_components"],
            "rmsecv": result["rmsecv"],
            "r2": result["r2_train"],
            "predictions": predictions,
        },
        report="\n".join(report_lines),
        raw_data=result,
    )
