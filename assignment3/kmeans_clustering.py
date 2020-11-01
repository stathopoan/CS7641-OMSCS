from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, homogeneity_score, completeness_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from NN_experiment import run_NN
from utils import plot_sse_curve


def perform_k_means_clustering(X, y, dataset, DR=False, alg=None, show_silhouette=True):
    algorithm = "kmeans"

    DR_FLAG = ""
    if DR:
        DR_FLAG = "YesDR"
    else:
        DR_FLAG = "NoDR"

    print("### Run Kmeans on  {} dataset DR: {} with alg: {} ###".format(dataset, DR_FLAG, alg))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9098, shuffle=True)

    # Run kmeans for different clusters and save sse metric
    n_clusters_list = range(2, 26, 2)
    sse = []
    homogeneity_scores = []
    completeness_scores = []
    silhouette_scores = []
    for n_clusters in n_clusters_list:
        clusterer = KMeans(n_clusters=n_clusters, random_state=12, max_iter=1000).fit(X_train)
        sse.append(clusterer.inertia_)

        homog_avg = homogeneity_score(clusterer.predict(X_test), y_test)
        homogeneity_scores.append(homog_avg)
        complet_avg = completeness_score(clusterer.predict(X_test), y_test)
        completeness_scores.append(complet_avg)
        silhouet_score = silhouette_score(X_train, clusterer.predict(X_train))
        silhouette_scores.append(silhouet_score)

    if alg is None:

        calc_metrics_curve(n_clusters_list, homogeneity_scores, completeness_scores, silhouette_scores, algorithm,
                           dataset,
                           filename="plots\\{}\\{}\\metrics_{}.png".format(dataset, DR_FLAG, algorithm))
        plot_sse_curve(sse, n_clusters_list, dataset,
                       filename="plots\\{}\\{}\\SSE_{}.png".format(dataset, DR_FLAG, algorithm))
    else:

        calc_metrics_curve(n_clusters_list, homogeneity_scores, completeness_scores, silhouette_scores, algorithm,
                           dataset,
                           filename="plots\\{}\\{}\\{}\\metrics_{}.png".format(dataset, DR_FLAG, alg, algorithm))
        plot_sse_curve(sse, n_clusters_list, dataset,
                       filename="plots\\{}\\{}\\{}\\SSE_{}.png".format(dataset, DR_FLAG, alg, algorithm))

    if show_silhouette:
        calc_silhouete_curve(n_clusters_list, X_train, algorithm, dataset, DR_FLAG, alg)

    if dataset == "heart" and DR is False:
        k_selected = 13

        clusterer = KMeans(n_clusters=k_selected, random_state=12, max_iter=1000).fit(X)
        labels = clusterer.labels_
        X_kmeans = labels.reshape(labels.shape[0], 1)
        run_NN(X_kmeans, y, alg=None, dataset=dataset, algorithm="kmeans")


# Reference: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
def calc_silhouete_curve(n_clusters_list, X, algorithm, dataset, DR_FLAG, alg):
    for n_clusters in n_clusters_list:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)

        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle("Silhouette analysis for {} clustering on sample data "
                     "with n_clusters = {}".format(algorithm, n_clusters),
                     fontsize=14, fontweight='bold')
        if alg is None:
            plt.savefig(
                "plots\\{}\\{}\\silhouette_n_clusters_{}_{}.png".format(dataset, DR_FLAG, n_clusters, algorithm))
        else:
            plt.savefig("plots\\{}\\{}\\{}\\silhouette_n_clusters_{}_{}.png".format(dataset, DR_FLAG, alg, n_clusters,
                                                                                    algorithm))
        plt.close()
    # plt.show()


def calc_metrics_curve(n_clusters_list, homogeneity_scores, completeness_scores, silhouette_scores, algorithm, dataset,
                       filename):
    plt.title("Metrics vs n_clusters - dataset: {} - algorithm: {}".format(dataset, algorithm))
    plt.xlabel(r"No of clusters")
    plt.ylabel("Metric")
    lw = 2
    plt.plot(n_clusters_list, homogeneity_scores, label="homogeneity", lw=lw)
    plt.plot(n_clusters_list, completeness_scores, label="completeness", lw=lw)
    plt.plot(n_clusters_list, silhouette_scores, label="silhouette", lw=lw)
    plt.legend(loc="best")
    # plt.savefig("plots\\{}\\{}\\metrics_{}.png".format(dataset, DR_FLAG, algorithm))
    plt.savefig(filename)
    plt.close()
