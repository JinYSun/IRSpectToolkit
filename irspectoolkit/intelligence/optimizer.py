"""
参数自动寻优 — 基于Optuna贝叶斯优化
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression


class SpectralOptimizer:
    """
    光谱分析参数自动寻优引擎
    
    自动搜索最优的预处理组合 + 模型超参数
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, task: str = "classification"):
        self.X = X
        self.y = y
        self.task = task

    def optimize_classification(
        self,
        models: list[str] | None = None,
        preprocessing_options: dict | None = None,
        n_trials: int = 50,
        cv_folds: int = 5,
    ) -> dict:
        """
        自动优化分类pipeline
        
        搜索空间: 预处理方法 + 模型类型 + 超参数
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("需要安装 optuna: pip install optuna")

        if models is None:
            models = ["knn", "svm", "pls_da"]

        def objective(trial):
            # 预处理选择
            do_snv = trial.suggest_categorical("snv", [True, False])
            do_derivative = trial.suggest_categorical("derivative", [True, False])
            deriv_order = trial.suggest_int("deriv_order", 1, 2) if do_derivative else 1
            sg_window = trial.suggest_int("sg_window", 5, 31, step=2)
            sg_poly = trial.suggest_int("sg_poly", 1, 3)

            X_processed = self.X.copy()
            if do_snv:
                from irspectoolkit.preprocessing.transform import snv
                X_processed = snv(X_processed)
            if do_derivative:
                from irspectoolkit.preprocessing.transform import savgol_derivative
                X_processed = savgol_derivative(X_processed, sg_window, sg_poly, deriv_order)

            # 模型选择
            model_name = trial.suggest_categorical("model", models)

            if model_name == "knn":
                n_neighbors = trial.suggest_int("n_neighbors", 3, 20)
                clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            elif model_name == "svm":
                C = trial.suggest_float("svm_C", 0.1, 100, log=True)
                kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
                clf = SVC(C=C, kernel=kernel, random_state=42)
            else:  # pls_da
                n_comp = trial.suggest_int("pls_n_comp", 2, 15)
                from irspectoolkit.analysis.classify import classify_pls_da
                # PLS-DA用特殊评估方式
                clf = PLSRegression(n_components=n_comp)

            # 交叉验证
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            if model_name == "pls_da":
                scores = cross_val_score(clf, X_processed, self.y.astype(float), cv=cv, scoring="r2")
            else:
                scores = cross_val_score(clf, X_processed, self.y, cv=cv, scoring="accuracy")

            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_trial
        return {
            "best_score": round(best.value, 4),
            "best_params": best.params,
            "n_trials": n_trials,
            "all_results": [
                {"params": t.params, "score": round(t.value, 4)}
                for t in study.trials
            ],
        }

    def optimize_regression(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
    ) -> dict:
        """自动优化回归pipeline"""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("需要安装 optuna: pip install optuna")

        def objective(trial):
            do_snv = trial.suggest_categorical("snv", [True, False])
            sg_window = trial.suggest_int("sg_window", 5, 31, step=2)
            sg_poly = trial.suggest_int("sg_poly", 1, 3)
            n_comp = trial.suggest_int("pls_n_comp", 2, 20)

            X_processed = self.X.copy()
            if do_snv:
                from irspectoolkit.preprocessing.transform import snv
                X_processed = snv(X_processed)
            from irspectoolkit.preprocessing.transform import savgol_derivative
            X_processed = savgol_derivative(X_processed, sg_window, sg_poly, 1)

            pls = PLSRegression(n_components=n_comp)
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            y_cv = cross_val_score(pls, X_processed, self.y, cv=cv, scoring="r2")
            return y_cv.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return {
            "best_score": round(study.best_value, 4),
            "best_params": study.best_params,
            "n_trials": n_trials,
        }
