"""定量回归分析"""
from __future__ import annotations
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score


def pls_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 10,
    cv_folds: int = 5,
) -> dict:
    """
    PLS回归 + 交叉验证自动选择最佳成分数
    
    Returns:
        最佳模型、RMSECV、R²、最佳成分数
    """
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    best_n = 1
    best_rmsecv = float("inf")

    rmsecv_list = []
    for n in range(1, n_components + 1):
        pls = PLSRegression(n_components=n)
        cv = KFold(n_splits=min(cv_folds, X.shape[0]), shuffle=True, random_state=42)
        y_cv = cross_val_predict(pls, X, y, cv=cv)
        rmsecv = np.sqrt(mean_squared_error(y, y_cv))
        rmsecv_list.append(rmsecv)

        if rmsecv < best_rmsecv:
            best_rmsecv = rmsecv
            best_n = n

    # 用最佳成分数训练最终模型
    best_pls = PLSRegression(n_components=best_n)
    best_pls.fit(X, y)
    y_pred = best_pls.predict(X).flatten()

    return {
        "model": best_pls,
        "best_n_components": best_n,
        "rmsecv": round(best_rmsecv, 4),
        "r2_train": round(r2_score(y, y_pred), 4),
        "rmsecv_curve": rmsecv_list,
        "predictions_train": y_pred,
    }


def pcr_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 10,
    cv_folds: int = 5,
) -> dict:
    """主成分回归 (PCR)"""
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    best_n = 1
    best_rmsecv = float("inf")

    for n in range(1, n_components + 1):
        pca = PCA(n_components=n)
        scores = pca.fit_transform(X)
        cv = KFold(n_splits=min(cv_folds, X.shape[0]), shuffle=True, random_state=42)
        y_cv = cross_val_predict(LinearRegression(), scores, y, cv=cv)
        rmsecv = np.sqrt(mean_squared_error(y, y_cv))

        if rmsecv < best_rmsecv:
            best_rmsecv = rmsecv
            best_n = n

    pca = PCA(n_components=best_n)
    scores = pca.fit_transform(X)
    lr = LinearRegression()
    lr.fit(scores, y)
    y_pred = lr.predict(scores)

    return {
        "model": (pca, lr),
        "best_n_components": best_n,
        "rmsecv": round(best_rmsecv, 4),
        "r2_train": round(r2_score(y, y_pred), 4),
    }


def svr_regression(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str = "rbf",
    C: float = 100.0,
    epsilon: float = 0.1,
) -> dict:
    """支持向量回归"""
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X, y)
    y_pred = model.predict(X)

    return {
        "model": model,
        "r2_train": round(r2_score(y, y_pred), 4),
        "rmse_train": round(np.sqrt(mean_squared_error(y, y_pred)), 4),
        "predictions_train": y_pred,
    }
