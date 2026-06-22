"""光谱库搜索 — 余弦相似度、相关系数、HQI"""
from __future__ import annotations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine


def cosine_search(
    query: np.ndarray,
    library: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    """余弦相似度搜索"""
    if query.ndim == 1:
        query = query.reshape(1, -1)
    sims = sk_cosine(query, library).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [{"index": int(i), "score": round(float(sims[i]), 4)} for i in top_indices]


def correlation_search(
    query: np.ndarray,
    library: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    """相关系数搜索"""
    if query.ndim == 1:
        query = query.flatten()
    corrs = []
    for i in range(library.shape[0]):
        corr = np.corrcoef(query, library[i])[0, 1]
        corrs.append(corr)
    corrs = np.array(corrs)
    top_indices = np.argsort(corrs)[::-1][:top_k]
    return [{"index": int(i), "score": round(float(corrs[i]), 4)} for i in top_indices]


def hqi_search(
    query: np.ndarray,
    library: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    """
    Hit Quality Index (HQI) — 综合打分
    
    HQI = 0.5 * 余弦相似度 + 0.5 * 相关系数
    """
    if query.ndim == 1:
        query_q = query.reshape(1, -1)
    else:
        query_q = query

    cos_scores = sk_cosine(query_q, library).flatten()
    corr_scores = np.array([
        np.corrcoef(query.flatten(), library[i])[0, 1]
        for i in range(library.shape[0])
    ])

    hqi = 0.5 * cos_scores + 0.5 * corr_scores
    top_indices = np.argsort(hqi)[::-1][:top_k]
    return [{"index": int(i), "score": round(float(hqi[i]), 4), "cosine": round(float(cos_scores[i]), 4), "correlation": round(float(corr_scores[i]), 4)} for i in top_indices]


class SpectralLibrary:
    """光谱库管理"""

    def __init__(self):
        self.spectra: list[np.ndarray] = []
        self.names: list[str] = []
        self.metadata: list[dict] = []

    def add(self, name: str, spectrum: np.ndarray, metadata: dict | None = None):
        self.spectra.append(spectrum.flatten())
        self.names.append(name)
        self.metadata.append(metadata or {})

    def add_batch(self, names: list[str], spectra: np.ndarray, metadata: list[dict] | None = None):
        for i, (name, spec) in enumerate(zip(names, spectra)):
            self.add(name, spec, metadata[i] if metadata else None)

    @property
    def array(self) -> np.ndarray:
        return np.array(self.spectra)

    def search(self, query: np.ndarray, method: str = "hqi", top_k: int = 5) -> list[dict]:
        """搜索光谱库"""
        lib_array = self.array
        if method == "cosine":
            results = cosine_search(query, lib_array, top_k)
        elif method == "correlation":
            results = correlation_search(query, lib_array, top_k)
        elif method == "hqi":
            results = hqi_search(query, lib_array, top_k)
        else:
            raise ValueError(f"不支持的搜索方法: {method}")

        for r in results:
            r["name"] = self.names[r["index"]]
            r["metadata"] = self.metadata[r["index"]]
        return results

    def __len__(self):
        return len(self.spectra)

    def __repr__(self):
        return f"SpectralLibrary({len(self)} spectra)"
