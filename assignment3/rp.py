from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.random_projection import GaussianRandomProjection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from EM_clustering import perform_em_clustering
from NN_experiment import run_NN
from kmeans_clustering import perform_k_means_clustering


def run_rp(X, y, dataset):
    DR_FLAG = "YesDR"
    alg = "rp"
    "### Running ΡΠ for dataset: {} ###".format(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9098, shuffle=True)

    n_components_list = range(3, 11, 1)
    repeats = 10
    rng = np.random.RandomState(42)
    sse = np.zeros((repeats, len(n_components_list)))

    for i in range(repeats):
        j = 0
        for n_components in n_components_list:
            grp = GaussianRandomProjection(n_components=n_components, random_state=rng)
            X_train_GRP = grp.fit_transform(X_train)

            A = np.linalg.pinv(grp.components_.T)  # pseudoinverse
            reconstructed = X_train_GRP.dot(A)
            rc_err = mean_squared_error(X_train, reconstructed)
            sse[i, j] = rc_err
            if i == 0:
                plot_2d_dimensions(X_train_GRP, y_train,
                                   filename="plots\\{}\\{}\\{}\\rp_visual_2D_{}_components.png".format(dataset, DR_FLAG,
                                                                                                       alg,
                                                                                                       n_components))
                plot_3d_dimensions(X_train_GRP, y_train,
                                   filename="plots\\{}\\{}\\{}\\rp_visual_3D_{}_components.png".format(dataset, DR_FLAG,
                                                                                                       alg,
                                                                                                       n_components))
            j += 1

    plot_sse_curve(sse, len(n_components_list), dataset,
                   filename="plots\\{}\\{}\\{}\\SSE_vs_n_components.png".format(dataset, DR_FLAG, alg))

    best_n_components = 2
    if dataset == "heart":
        best_n_components = 6
    elif dataset == "wine":
        best_n_components = 8

    rp_best_model = GaussianRandomProjection(n_components=best_n_components, random_state=rng)
    rp_best_model.fit(X_train)
    perform_k_means_clustering(rp_best_model.transform(X), y, dataset, DR=True, alg=alg)
    perform_em_clustering(rp_best_model.transform(X), y, dataset, DR=True, alg=alg)

    if dataset == "heart":
        run_NN(rp_best_model.transform(X), y, alg, dataset, algorithm=None)
    # grp = GaussianRandomProjection(random_state=rng)
    # clusterer = KMeans(random_state=12, max_iter=1000)
    # pipe = Pipeline(steps=[('grp', grp), ('kmeans', clusterer)])
    # param_grid = {
    #     'grp__n_components': range(3, 11, 1),
    #     'kmeans__n_clusters': range(2, 26, 2),
    # }
    # search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    # search.fit(X_train, y_train)
    # print(search.best_params_)

def plot_sse_curve(sse, n_components, dataset, filename):
    # print ("Start validation curve")

    sse_mean = np.mean(sse, axis=0)
    sse_std = np.std(sse, axis=0)

    plt.title("SSE Curve - Dataset: {}".format(dataset))
    plt.xlabel("n_components")
    plt.ylabel("SSE")
    # plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(range(3, n_components + 3, 1), sse_mean, label="SSE score", color="darkorange", lw=lw)
    plt.fill_between(range(3, n_components + 3, 1), sse_mean - sse_std, sse_mean + sse_std, alpha=0.2,
                     color="darkorange", lw=lw)

    plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()


def plot_2d_dimensions(components, y_train, filename):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('component 1', fontsize=15)
    ax.set_ylabel('component 2', fontsize=15)
    ax.set_title('2 component RP', fontsize=20)
    targets = [0, 1]
    colors = ['g', 'r']
    for target, color in zip(targets, colors):
        indicesToKeep = y_train == target
        ax.scatter(components[indicesToKeep, 0]
                   , components[indicesToKeep, 1]
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    # plt.show()
    plt.savefig(filename)
    plt.close()


def plot_3d_dimensions(components, y_train, filename):
    colors = ['g', 'r']
    targets = [0, 1]
    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    for target, color in zip(targets, colors):
        indicesToKeep = y_train == target
        ax.scatter(
            xs=components[indicesToKeep, 0],
            ys=components[indicesToKeep, 1],
            zs=components[indicesToKeep, 2],
            c=color,
            cmap='tab10'
        )
    ax.set_xlabel('comp-one')
    ax.set_ylabel('comp-two')
    ax.set_zlabel('comp-three')
    # plt.show()
    plt.savefig(filename)
    plt.close()
