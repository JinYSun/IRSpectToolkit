"""降维方法"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def reduce_pca(spectra: np.ndarray, n_components: int = 3) -> dict:
    """PCA降维"""
    n_components = min(n_components, spectra.shape[0] - 1, spectra.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(spectra)
    return {
        "scores": scores,
        "loadings": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        "model": pca,
    }


def reduce_pls(spectra: np.ndarray, y: np.ndarray, n_components: int = 3) -> dict:
    """PLS降维 (有监督)"""
    pls = PLSRegression(n_components=n_components)
    pls.fit(spectra, y)
    scores = pls.x_scores_
    return {
        "scores": scores,
        "loadings": pls.x_loadings_,
        "model": pls,
    }


def reduce_tsne(spectra: np.ndarray, n_components: int = 2, perplexity: float = 30.0) -> dict:
    """t-SNE降维 (可视化)"""
    n_components = min(n_components, spectra.shape[0] - 1)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    embedding = tsne.fit_transform(spectra)
    return {"embedding": embedding, "model": tsne}


def reduce_umap(spectra: np.ndarray, n_components: int = 2, n_neighbors: int = 15) -> dict:
    """UMAP降维 (需要umap-learn)"""
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
        embedding = reducer.fit_transform(spectra)
        return {"embedding": embedding, "model": reducer}
    except ImportError:
        raise ImportError("需要安装 umap-learn: pip install umap-learn")


def reduce_lda(spectra: np.ndarray, labels: np.ndarray, n_components: int | None = None) -> dict:
    """LDA降维 (有监督)"""
    n_classes = len(np.unique(labels))
    if n_components is None:
        n_components = min(n_classes - 1, spectra.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    scores = lda.fit_transform(spectra, labels)
    return {"scores": scores, "scalings": lda.scalings_, "model": lda}
