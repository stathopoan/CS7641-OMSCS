from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from EM_clustering import perform_em_clustering
from NN_experiment import run_NN
from kmeans_clustering import perform_k_means_clustering


def run_lda(X, y, dataset):
    DR_FLAG = "YesDR"
    alg = "lda"
    "### Running LDA for dataset: {} ###".format(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9098, shuffle=True)

    n_components =1
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train, y_train)

    plot_1d_dimensions(X_train_lda, y_train,
                       filename="plots\\{}\\{}\\{}\\rp_visual_2D_{}_components.png".format(dataset, DR_FLAG,
                                                                                           alg,
                                                                                           n_components))

    perform_k_means_clustering(lda.transform(X), y, dataset, DR=True, alg=alg, show_silhouette=False)
    perform_em_clustering(lda.transform(X), y, dataset, DR=True, alg=alg, show_silhouette=False)

    if dataset == "heart":
        run_NN(lda.transform(X), y, alg, dataset, algorithm=None)


def plot_sse_curve(sse, components, dataset, filename):
    plt.title("SSE vs n_components - dataset: {}".format(dataset))
    plt.xlabel(r"No of components")
    plt.ylabel("SSE")
    lw = 2
    plt.plot(components, sse, lw=lw)
    # plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def plot_1d_dimensions(components, y_train, filename):
    fig = plt.figure(figsize=(8, 8))
    targets = [0, 1]
    colors = ['g', 'r']
    for target, color in zip(targets, colors):
        indicesToKeep = y_train == target
        x = components[indicesToKeep, 0]
        y = np.zeros(len(x))
        plt.scatter(x,y, color=color, label=target)
    plt.savefig(filename)
    plt.close()