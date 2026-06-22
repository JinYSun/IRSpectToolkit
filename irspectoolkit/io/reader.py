"""
多格式光谱数据读取
支持: CSV, Excel, SPC, JDX, SPA
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field


@dataclass
class SpectralData:
    """统一的光谱数据容器"""
    spectra: np.ndarray          # shape: (n_samples, n_wavelengths)
    wavelengths: np.ndarray      # shape: (n_wavelengths,)
    labels: Optional[np.ndarray] = None  # shape: (n_samples,)
    sample_ids: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    source_file: str = ""

    @property
    def n_samples(self) -> int:
        return self.spectra.shape[0]

    @property
    def n_wavelengths(self) -> int:
        return self.spectra.shape[1]

    @property
    def wavenumber_range(self) -> tuple:
        return (float(self.wavelengths.min()), float(self.wavelengths.max()))

    def __repr__(self):
        return (
            f"SpectralData({self.n_samples} samples × {self.n_wavelengths} points, "
            f"range: {self.wavenumber_range[0]:.0f}-{self.wavenumber_range[1]:.0f})"
        )


class SpectralReader:
    """光谱数据读取器"""

    @staticmethod
    def from_csv(
        filepath: str,
        wavelength_col: int = 0,
        label_col: Optional[int] = None,
        header: bool = True,
        transpose: bool = False,
        delimiter: str = ",",
    ) -> SpectralData:
        """
        从CSV文件读取光谱

        支持两种格式:
        1. 行=波长, 列=样本 (transpose=False)
        2. 行=样本, 列=波长 (transpose=True)
        """
        df = pd.read_csv(filepath, header=0 if header else None, delimiter=delimiter)

        if transpose:
            df = df.T

        # 提取波长
        wavelengths = df.iloc[:, wavelength_col].values.astype(float)
        if wavelength_col == 0:
            spectra_df = df.iloc[:, 1:]
        else:
            spectra_df = df.drop(df.columns[wavelength_col], axis=1)

        spectra = spectra_df.values.astype(float).T  # (n_samples, n_wavelengths)

        # 提取标签
        labels = None
        if label_col is not None:
            labels = df.iloc[:, label_col].values

        sample_ids = [f"sample_{i}" for i in range(spectra.shape[0])]

        return SpectralData(
            spectra=spectra,
            wavelengths=wavelengths,
            labels=labels,
            sample_ids=sample_ids,
            source_file=str(filepath),
        )

    @staticmethod
    def from_excel(
        filepath: str,
        sheet_name: Union[str, int] = 0,
        wavelength_col: int = 0,
    ) -> SpectralData:
        """从Excel文件读取光谱"""
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        wavelengths = df.iloc[:, wavelength_col].values.astype(float)
        spectra = df.iloc[:, wavelength_col + 1:].values.astype(float).T
        sample_ids = [f"sample_{i}" for i in range(spectra.shape[0])]

        return SpectralData(
            spectra=spectra,
            wavelengths=wavelengths,
            sample_ids=sample_ids,
            source_file=str(filepath),
        )

    @staticmethod
    def from_numpy(
        spectra: np.ndarray,
        wavelengths: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> SpectralData:
        """从numpy数组创建SpectralData"""
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        sample_ids = [f"sample_{i}" for i in range(spectra.shape[0])]
        return SpectralData(
            spectra=spectra,
            wavelengths=wavelengths,
            labels=labels,
            sample_ids=sample_ids,
        )

    @staticmethod
    def from_spc(filepath: str) -> SpectralData:
        """
        从SPC文件读取 (需要spectrochempy)
        SPC是Grams/Thermo Galactic通用光谱格式
        """
        try:
            import spectrochempy as scp
            nd = scp.read_spc(filepath)
            spectra = nd.data
            wavelengths = nd.x.data
            if spectra.ndim == 1:
                spectra = spectra.reshape(1, -1)
            return SpectralData(
                spectra=spectra,
                wavelengths=wavelengths,
                sample_ids=[f"sample_{i}" for i in range(spectra.shape[0])],
                source_file=str(filepath),
                metadata={"format": "spc"},
            )
        except ImportError:
            raise ImportError("需要安装 spectrochempy: pip install spectrochempy")

    @staticmethod
    def from_jdx(filepath: str) -> SpectralData:
        """从JCAMP-DX文件读取"""
        try:
            import spectrochempy as scp
            nd = scp.read_jdx(filepath)
            spectra = nd.data
            wavelengths = nd.x.data
            if spectra.ndim == 1:
                spectra = spectra.reshape(1, -1)
            return SpectralData(
                spectra=spectra,
                wavelengths=wavelengths,
                source_file=str(filepath),
                metadata={"format": "jdx"},
            )
        except ImportError:
            raise ImportError("需要安装 spectrochempy: pip install spectrochempy")


# 便捷函数
def read_csv(filepath: str, **kwargs) -> SpectralData:
    return SpectralReader.from_csv(filepath, **kwargs)

def read_spectra(filepath: str, **kwargs) -> SpectralData:
    """自动识别格式并读取"""
    ext = Path(filepath).suffix.lower()
    reader = SpectralReader()
    if ext == ".csv":
        return reader.from_csv(filepath, **kwargs)
    elif ext in (".xls", ".xlsx"):
        return reader.from_excel(filepath, **kwargs)
    elif ext == ".spc":
        return reader.from_spc(filepath)
    elif ext in (".jdx", ".dx"):
        return reader.from_jdx(filepath)
    elif ext in (".npy", ".npz"):
        data = np.load(filepath, allow_pickle=True)
        return reader.from_numpy(data["spectra"], data["wavelengths"])
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
