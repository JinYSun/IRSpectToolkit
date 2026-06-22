# analysis package
from .dimension import reduce_pca, reduce_pls, reduce_tsne, reduce_umap, reduce_lda
from .classify import simca_classify, classify_knn, classify_svm, classify_rf, classify_pls_da
from .regression import pls_regression, pcr_regression, svr_regression
from .peak import detect_peaks, assign_functional_groups, integrate_peak_area
from .library import cosine_search, correlation_search, hqi_search, SpectralLibrary
