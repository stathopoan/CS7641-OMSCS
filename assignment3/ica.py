from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import log

from EM_clustering import perform_em_clustering
from NN_experiment import run_NN
from kmeans_clustering import perform_k_means_clustering
from utils import plot_sse_curve


def run_ica(X, y, dataset):
    DR_FLAG = "YesDR"
    alg = "ica"
    "### Running ICA for dataset: {} ###".format(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9098, shuffle=True)

    n_components_list = range(2, 11, 1)
    sse = []

    for n_components in n_components_list:

        ica = FastICA(n_components=n_components, random_state=44)

        S_ = ica.fit_transform(X)  # Reconstruct signals
        kurt = kurtosisMatrix(S_)
        print ("no of components: {} - kurtosis: {}".format(n_components, kurt))
        A_ = ica.mixing_  # Get estimated mixing matrix
        X_reconstructed = np.dot(S_, A_.T) + ica.mean_
        ss_err = mean_squared_error(X, X_reconstructed)
        sse.append(ss_err)

    plot_sse_curve(sse, n_components_list, dataset,
                   filename="plots\\{}\\{}\\{}\\SSE_vs_n_components.png".format(dataset, DR_FLAG, alg))


    ica = FastICA(n_components=2, random_state=44)
    ica_components = ica.fit_transform(X_train)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Independent Component 1', fontsize=15)
    ax.set_ylabel('Independent Component 2', fontsize=15)
    ax.set_title('2 component ICA', fontsize=20)
    targets = [0, 1]
    colors = ['g', 'r']
    for target, color in zip(targets, colors):
        indicesToKeep = y_train == target
        ax.scatter(ica_components[indicesToKeep, 0]
                   , ica_components[indicesToKeep, 1]
                   , c=color
                   , s=50)
    # axis = -ica.mixing_
    # for color, axis in zip(colors, axis):
    #     x_axis, y_axis = axis
    #     plt.plot(0.1 * x_axis, 0.1 * y_axis, linewidth=2, color=color)
    #     plt.quiver((0, 0), (0, 0), x_axis, y_axis, zorder=11, width=0.01,
    #            scale=6, color=color)
    ax.legend(targets)
    ax.grid()
    # plt.show()
    plt.savefig("plots\\{}\\{}\\{}\\ica_visual_2_components.png".format(dataset, DR_FLAG, alg))
    plt.close()

    ica = FastICA(n_components=3, random_state=44)
    ica_components = ica.fit_transform(X_train)
    colors = ['g', 'r']
    targets = [0, 1]
    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    for target, color in zip(targets, colors):
        indicesToKeep = y_train == target
        ax.scatter(
            xs=ica_components[indicesToKeep, 0],
            ys=ica_components[indicesToKeep, 1],
            zs=ica_components[indicesToKeep, 2],
            c=color,
            cmap='tab10'
        )
    ax.set_xlabel('ica-one')
    ax.set_ylabel('ica-two')
    ax.set_zlabel('ica-three')
    ax.legend(targets)
    # plt.show()
    plt.savefig("plots\\{}\\{}\\{}\\ica_visual_3_components.png".format(dataset, DR_FLAG, alg))
    plt.close()

    best_n_components = 2

    if dataset == "heart":
        best_n_components = 6
    elif dataset == "wine":
        best_n_components = 7

    ica_best_model = FastICA(n_components=best_n_components, random_state=44)
    ica_best_model.fit(X_train)

    X_train_ica = ica_best_model.transform(X_train)
    X_projected = ica_best_model.inverse_transform(X_train_ica)
    proj_loss = ((X_train - X_projected) ** 2).mean()

    print("Total projection loss: {} ".format(proj_loss))

    perform_k_means_clustering(ica_best_model.transform(X), y, dataset, DR=True, alg=alg)
    perform_em_clustering(ica_best_model.transform(X), y, dataset, DR=True, alg=alg)

    if dataset == "heart":
        run_NN(ica_best_model.transform(X), y, alg, dataset, algorithm=None)


# Reference: https://towardsdatascience.com/separating-mixed-signals-with-independent-component-analysis-38205188f2f4
def kurtosis(x):
    n = np.shape(x)[0]
    mean = np.sum((x**1)/n) # Calculate the mean
    var = np.sum((x-mean)**2)/n # Calculate the variance
    skew = np.sum((x-mean)**3)/n # Calculate the skewness
    kurt = np.sum((x-mean)**4)/n # Calculate the kurtosis
    kurt = kurt/(var**2)-3
    return kurt, skew, var, mean

def kurtosisMatrix(x):
    result = []
    for i in range(0,x.shape[1],1):
        kurt, _, _, _ = kurtosis(x[:,i])
        result.append(kurt)
    return np.array(result)
