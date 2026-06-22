# IRSpecToolkit

**A Modular Automatic Infrared Spectroscopy Analysis Framework Integrating Classical Chemometrics, Deep Learning and Knowledge-Guided Optimization**

IRSpecToolkit is a modular Python framework dedicated to infrared spectroscopy (NIR / MIR / FTIR / ATR) analysis. It integrates full-spectrum workflows including preprocessing, baseline correction, dimensionality reduction, classification, regression, peak analysis, spectral library matching, deep learning, data augmentation, domain adaptation, and automatic hyperparameter optimization, while providing high-level scene-oriented Skill APIs tailored to real-world industrial and research scenarios.

## Table of Contents

- [Core Features](#core-features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Module Overview](#module-overview)
- [Examples](#examples)
- [Dependencies](#dependencies)
- [Optional Dependencies](#optional-dependencies)
- [Citation](#citation)

## Core Features

| Feature | Description |
|---------|-------------|
| Unified Toolkit for Full Spectral Bands | Full coverage of NIR / MIR / FTIR / ATR with unified application programming interfaces |
| 50+ Baseline Correction Algorithms | ALS, airPLS, SNIP, modPoly, arPLS and other mainstream baseline correction methods |
| Automatic Hyperparameter Optimization | Bayesian optimization powered by Optuna for joint tuning of preprocessing pipelines and model hyperparameters |
| Hybrid Classical & Deep Learning | Combines traditional chemometric methods (SIMCA/KNN/SVM/RF) and 1D-CNN to balance inference speed and predictive accuracy |
| Knowledge Injection Engine | Built-in functional group – characteristic peak mapping database for automatic peak assignment and annotation |
| Multi-Dimensional Visualization | Spectral overlay plots, PCA score plots, baseline comparison charts, peak attribution diagrams, and more |
| Scene-Oriented Skill Interfaces | One-click pipelines for rapid substance identification, authenticity verification, concentration quantification, anomaly screening, and functional group scanning |
| Spectral Data Augmentation | Noise injection, wavelength jitter, spectral Mixup and mixture generation to mitigate small-sample limitations |
| Spectral Quality Assessment | Signal-to-noise ratio estimation, repeatability evaluation and dataset integrity inspection |
| Cross-Instrument Domain Adaptation | TCA, CORAL and spectral alignment algorithms to enable model transfer across different spectrometers |
| Multi-Spectral Modal Fusion | Early and late fusion strategies for joint analysis of multi-source spectral datasets |

## Project Structure

```
ir-spectral-toolkit/
├── irspectoolkit/                # Core source package
│   ├── io/                       # Multi-format spectral data I/O
│   ├── quality/                  # Spectral quality evaluation (SNR/repeatability/integrity)
│   ├── preprocessing/            # Preprocessing pipelines (SNV/MSC/derivative/smoothing/baseline/outlier detection)
│   ├── augmentation/             # Spectral data augmentation modules
│   ├── analysis/                 # Chemometric analysis (dimensionality reduction/classification/regression/peak detection/library search)
│   ├── deeplearning/             # Deep learning modules (1D-CNN)
│   ├── transfer/                 # Domain adaptation algorithms (TCA/spectral alignment)
│   ├── fusion/                   # Multi-modal spectral fusion
│   ├── intelligence/             # Intelligent engine (hyperparameter optimizer / spectral knowledge base)
│   ├── visualization/            # Plotting and visualization utilities
│   └── skills/                   # Encapsulated scene-oriented analysis skills
├── examples/                     # Demo scripts for all core workflows
├── requirements.txt              # Mandatory dependency list
└── README.md
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### 5-Minute Minimal Demo

```python
import numpy as np
from irspectoolkit.preprocessing.transform import snv, savgol_derivative
from irspectoolkit.preprocessing.baseline import baseline_als
from irspectoolkit.analysis.dimension import reduce_pca
from irspectoolkit.analysis.peak import detect_peaks, assign_functional_groups
from irspectoolkit.skills import rapid_identify
from irspectoolkit.analysis.library import SpectralLibrary

# Step 1: Spectral preprocessing
spectra_snv = snv(spectra)
spectra_bl, baselines = baseline_als(spectra, lam=1e6, p=0.01)
spectra_deriv = savgol_derivative(spectra_snv, window_length=11, polyorder=2)

# Step 2: PCA dimensionality reduction
pca_result = reduce_pca(spectra_deriv, n_components=3)

# Step 3: Peak detection and functional group assignment
peaks = detect_peaks(mean_spectrum, wavelengths)
assignments = assign_functional_groups(peaks)

# Step 4: One-click rapid substance identification via encapsulated Skill
result = rapid_identify(query_spectrum, library)
print(result.conclusion, result.confidence)
```

## Module Overview

### 📥 io — Data I/O

Read multi-format spectral files including CSV, SPA, SPC, JCAMP, etc.

### 🔍 quality — Spectral Quality Evaluation

- `snr_estimate` — Signal-to-noise ratio quantification
- `repeatability_check` — Intra-batch measurement repeatability assessment
- `completeness_check` — Missing value and data integrity inspection
- `spectral_quality_score` — Comprehensive unified quality scoring metric

### ⚙️ preprocessing — Preprocessing Pipeline

#### Spectral Transformation

- `snv` — Standard Normal Variate transformation
- `msc` — Multiplicative Scatter Correction
- `savgol_smooth` — Savitzky-Golay smoothing filter
- `savgol_derivative` — Savitzky-Golay first/second derivative
- `minmax_normalize` — Min-max normalization
- `wavelength_cut` — Custom spectral band truncation

#### Baseline Correction

- `baseline_als` — Asymmetric Least Squares baseline fitting
- `baseline_airpls` — Adaptive Iteratively Reweighted PLS
- `baseline_snip` — Statistics-sensitive Non-linear Iterative Peak-clipping
- `baseline_modpoly` — Modified polynomial baseline fitting
- `baseline_arpls` — Asymmetric Reweighted Penalized Least Squares

#### Outlier Detection

- `detect_outliers_mahalanobis` — Mahalanobis distance outlier screening
- `detect_outliers_lof` — Local Outlier Factor
- `detect_outliers_pca_q` — PCA Q-statistic anomaly detection
- `detect_outliers_isolation_forest` — Isolation Forest for spectral outliers

### 📊 analysis — Chemometric Analysis

#### Dimensionality Reduction

- `reduce_pca` — Principal Component Analysis
- `reduce_pls` — Partial Least Squares
- `reduce_tsne` — t-distributed Stochastic Neighbor Embedding
- `reduce_umap` — Uniform Manifold Approximation and Projection
- `reduce_lda` — Linear Discriminant Analysis

#### Classification

- `simca_classify` — Soft Independent Modeling of Class Analogy
- `classify_knn` — K-Nearest Neighbors
- `classify_svm` — Support Vector Machine
- `classify_rf` — Random Forest
- `classify_pls_da` — Partial Least Squares Discriminant Analysis

#### Regression

- `pls_regression` — Partial Least Squares Regression
- `pcr_regression` — Principal Component Regression
- `svr_regression` — Support Vector Regression

#### Peak Analysis

- `detect_peaks` — Automatic characteristic peak localization
- `assign_functional_groups` — Knowledge-based functional group peak assignment
- `integrate_peak_area` — Quantitative peak area integration

#### Spectral Library Matching

- `cosine_search` — Cosine similarity matching
- `correlation_search` — Pearson correlation coefficient matching
- `hqi_search` — Hit Quality Index matching
- `SpectralLibrary` — Object-oriented spectral library management class

### 🧠 deeplearning — Deep Learning Submodule

- `CNN1DClassifier` — 1D Convolutional Neural Network for spectral classification

### 🚀 augmentation — Spectral Data Augmentation

- `add_noise` — Gaussian random noise injection
- `wavelength_jitter` — Wavelength coordinate random perturbation
- `spectral_mixup` — Spectral Mixup augmentation
- `generate_mixture` — Artificial mixed spectrum generation
- `augment_dataset` — Batch automatic dataset augmentation pipeline

### 🔄 transfer — Domain Adaptation

- `spectral_align` — Cross-instrument spectral signal alignment
- `tca_transform` — Transfer Component Analysis

### 🔗 fusion — Multi-Modal Spectral Fusion

- `early_fusion` — Feature-level early fusion of multi-source spectra
- `late_fusion` — Prediction-level late fusion
- `align_wavelengths` — Wavelength axis unification for heterogeneous spectra

### 🤖 intelligence — Intelligent Optimization Engine

- `SpectralOptimizer` — Optuna-powered Bayesian joint hyperparameter search for preprocessing + model training
- `SpectralKnowledgeBase` — Persistent spectral knowledge base storing functional group – characteristic peak mappings

### 📈 visualization — Plotting Utilities

- `plot_spectra` — Overlaid raw/preprocessed spectral curves
- `plot_pca_scores` — PCA 2D/3D score scatter plots
- `plot_baseline_comparison` — Visual comparison of raw spectra and fitted baselines
- `plot_peak_assignment` — Annotated spectrum with labeled functional group peaks
- `plot_confusion_matrix` — Classification confusion matrix heatmap
- `plot_regression` — Predicted vs. reference concentration regression plots

### 🎯 skills — Encapsulated Scene-Oriented Analysis Pipelines

- `rapid_identify` — Fast qualitative substance identification
- `authenticity_check` — Product authenticity and counterfeit screening
- `batch_consistency` — Inter-batch spectral consistency evaluation
- `concentration_predict` — Quantitative component concentration prediction
- `anomaly_screen` — Production abnormal sample screening
- `functional_group_scan` — Full-spectrum functional group detection and profiling

## Examples

| Demo Script | Description |
|-------------|-------------|
| `demo_quickstart.py` | End-to-end standard workflow: Preprocessing → PCA → Classification → Peak Assignment → Library Matching → Scene Skill Inference |
| `demo_auto_optimize.py` | Automated pipeline hyperparameter tuning with Optuna Bayesian optimization |
| `demo_concentration_pls.py` | Quantitative concentration prediction via PLS / PCR / SVR regression |
| `demo_pharma_authenticity.py` | Pharmaceutical raw material authenticity verification workflow |
| `demo_deep_learning.py` | 1D-CNN deep learning classification for spectral datasets |
| `demo_knowledge_peak.py` | Knowledge-guided automatic functional group peak attribution |
| `demo_batch_consistency.py` | Inter-batch product consistency quality inspection pipeline |

### Run Demo Scripts

```bash
cd examples
python demo_quickstart.py
```

## Dependencies

| Package | Minimum Version | Purpose |
|---------|:---------------:|---------|
| numpy | 1.24 | Core numerical array computation |
| pandas | 2.0 | Tabular spectral dataset manipulation |
| scipy | 1.10 | Advanced scientific computing & signal processing |
| scikit-learn | 1.3 | Traditional machine learning algorithms |
| matplotlib | 3.7 | Static spectral visualization plotting |

## Optional Dependencies

| Package | Minimum Version | Purpose |
|---------|:---------------:|---------|
| chemotools | 0.1 | Supplementary chemometric preprocessing tools |
| pybaselines | 1.0 | Extended baseline correction algorithm library |
| spectrochempy | 0.4 | Advanced spectral chemometric utilities |
| torch | 2.0 | Backend for 1D-CNN deep learning models |
| umap-learn | 0.5 | UMAP non-linear dimensionality reduction |
| optuna | 3.0 | Bayesian hyperparameter optimization framework |
| xgboost | 2.0 | Gradient boosting classification & regression models |

## Note
It was edited by OpenClaw, Please double check before using!!!
## Citation

If you use IRSpecToolkit in your research work, please cite the following:

[> IRSpecToolkit: A Modular Python Framework for Automated Infrared Spectral Analysis Integrating Classical Chemometrics, Deep Learning, and Knowledge-Guided Optimization
](https://github.com/JinYSun/IRSpectToolkit)

## License

This open-source project is distributed under the MIT License.
