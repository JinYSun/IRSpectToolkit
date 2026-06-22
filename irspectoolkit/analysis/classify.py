"""定性分类方法"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report


def simca_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 5,
    threshold: float = 0.95,
) -> dict:
    """
    SIMCA分类 — 基于PCA子空间距离
    
    每个类别建立独立PCA模型，测试样本到各类子空间的残差决定归属
    """
    classes = np.unique(y_train)
    class_models = {}

    for cls in classes:
        X_cls = X_train[y_train == cls]
        n_comp = min(n_components, X_cls.shape[0] - 1, X_cls.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(X_cls)
        class_models[cls] = pca

    # 对测试集计算到各类的距离
    predictions = []
    distances = {cls: [] for cls in classes}

    for x in X_test.reshape(1, -1) if X_test.ndim == 1 else X_test:
        class_dist = {}
        for cls, pca in class_models.items():
            projected = pca.transform(x.reshape(1, -1))
            reconstructed = pca.inverse_transform(projected)
            residual = np.sum((x - reconstructed) ** 2)
            class_dist[cls] = residual
            distances[cls].append(residual)

        predictions.append(min(class_dist, key=class_dist.get))

    return {
        "predictions": np.array(predictions),
        "distances": distances,
        "class_models": class_models,
    }


def classify_knn(X_train, y_train, X_test, n_neighbors: int = 5) -> dict:
    """KNN分类"""
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return {"predictions": predictions, "model": clf, "proba": clf.predict_proba(X_test)}


def classify_svm(X_train, y_train, X_test, kernel: str = "rbf", C: float = 1.0) -> dict:
    """SVM分类"""
    clf = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return {"predictions": predictions, "model": clf, "proba": clf.predict_proba(X_test)}


def classify_rf(X_train, y_train, X_test, n_estimators: int = 100) -> dict:
    """随机森林分类"""
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return {"predictions": predictions, "model": clf, "proba": clf.predict_proba(X_test), "importance": clf.feature_importances_}


def classify_pls_da(X_train, y_train, X_test, n_components: int = 5) -> dict:
    """PLS-DA判别分析"""
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    y_pred_continuous = pls.predict(X_test)
    predictions = np.round(y_pred_continuous).astype(int).flatten()
    predictions = np.clip(predictions, int(y_train.min()), int(y_train.max()))
    return {"predictions": predictions, "model": pls, "raw_scores": y_pred_continuous}
