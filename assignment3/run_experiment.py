from EM_clustering import perform_em_clustering
from explore_data import explore_data_heart_disease, explore_data_wine
from ica import run_ica
from kmeans_clustering import perform_k_means_clustering
from lda import run_lda
from pca import run_pca
from rp import run_rp


def start_procedure():
    X_heart, y_heart = explore_data_heart_disease()
    X_wine, y_wine = explore_data_wine()

    perform_k_means_clustering(X_wine, y_wine, "wine")
    perform_em_clustering(X_wine, y_wine, "wine")

    perform_k_means_clustering(X_heart, y_heart, "heart")
    perform_em_clustering(X_heart, y_heart, "heart")

    run_pca(X_heart, y_heart, "heart")
    run_pca(X_wine, y_wine, "wine")

    run_ica(X_heart, y_heart, "heart")
    run_ica(X_wine, y_wine, "wine")

    run_rp(X_heart, y_heart, "heart")
    run_rp(X_wine, y_wine, "wine")

    run_lda(X_heart, y_heart, "heart")
    run_lda(X_wine, y_wine, "wine")


if __name__ == '__main__':
    start_procedure()