from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from EM_clustering import perform_em_clustering
from NN_experiment import run_NN
from kmeans_clustering import perform_k_means_clustering


def run_pca(X, y, dataset):
    DR_FLAG = "YesDR"
    alg = "pca"
    "### Running PCA for dataset: {} ###".format(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9098, shuffle=True)

    pca = PCA(n_components=10)
    principal_components = pca.fit_transform(X_train)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    # plt.show()
    plt.savefig("plots\\{}\\{}\\{}\\pca_variance_vs_components.png".format(dataset, DR_FLAG, alg))
    plt.close()

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_train)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [0, 1]
    colors = ['g', 'r']
    for target, color in zip(targets, colors):
        indicesToKeep = y_train == target
        ax.scatter(principal_components[indicesToKeep, 0]
                   , principal_components[indicesToKeep, 1]
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    # plt.show()
    plt.savefig("plots\\{}\\{}\\{}\\pca_visual_2_components.png".format(dataset, DR_FLAG, alg))
    plt.close()

    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(X_train)
    colors = ['g', 'r']
    targets = [0, 1]
    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    for target, color in zip(targets, colors):
        indicesToKeep = y_train == target
        ax.scatter(
            xs=principal_components[indicesToKeep, 0],
            ys=principal_components[indicesToKeep, 1],
            zs=principal_components[indicesToKeep, 2],
            c=color,
            cmap='tab10'
        )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    # plt.show()
    plt.savefig("plots\\{}\\{}\\{}\\pca_visual_3_components.png".format(dataset, DR_FLAG, alg))
    plt.close()

    best_n_components = 2

    if dataset == "heart":
        best_n_components = 6
    elif dataset == "wine":
        best_n_components = 7



    pca_best_model = PCA(n_components=best_n_components)
    pca_best_model.fit(X_train)

    print("Variance captured for every pca component: {}".format(pca_best_model.explained_variance_ratio_))
    print("Total variance captured: {} ".format(np.sum(pca_best_model.explained_variance_ratio_)))

    calc_eigenvalues_curve(pca_best_model.explained_variance_, dataset, filename="plots\\{}\\{}\\{}\\pca_eigenvalues_dist.png".format(dataset, DR_FLAG, alg))

    X_train_pca = pca_best_model.transform(X_train)
    X_projected = pca_best_model.inverse_transform(X_train_pca)
    proj_loss = ((X_train - X_projected) ** 2).mean()

    print("Total projection loss: {} ".format(proj_loss))

    perform_k_means_clustering(pca_best_model.transform(X), y, dataset, DR=True, alg=alg)
    perform_em_clustering(pca_best_model.transform(X), y, dataset, DR=True, alg=alg)

    if dataset == "heart":
        run_NN(pca_best_model.transform(X), y, alg, dataset, algorithm=None)


def calc_eigenvalues_curve(eigenvalues, dataset, filename):
    plt.title("Distribution of PCA eigenvalues - dataset: {}".format(dataset))
    plt.xlabel(r"Eigen Values")
    plt.ylabel("Variance")
    lw = 2
    plt.plot(eigenvalues, label="EigenValues", lw=lw)
    # plt.legend(loc="best")
    # plt.savefig("plots\\{}\\{}\\metrics_{}.png".format(dataset, DR_FLAG, algorithm))
    plt.savefig(filename)
    plt.close()


