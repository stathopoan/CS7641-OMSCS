import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_white_wine_data():
    input = './data/winequality-white.csv'
    df = pd.read_csv(input, header=0, sep=';')
    # dataset = np.loadtxt(input, delimiter=';')
    return df


def load_heart_disease_data():
    input = './data/reprocessed.hungarian.data'
    names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
             "thal", "target"]
    df = pd.read_csv(input, sep=" ", header=None, names=names)
    return df


def plot_sse_curve(sse, clusters, dataset, filename):
    plt.title("SSE vs n_clusters - dataset: {}".format(dataset))
    plt.xlabel(r"No of clusters")
    plt.ylabel("SSE")
    lw = 2
    plt.plot(clusters, sse, lw=lw)
    # plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def plot_lle_curve(sse, clusters, dataset, filename):
    plt.title("Log-Likelihood (LL) vs n_clusters - dataset: {}".format(dataset))
    plt.xlabel(r"No of clusters")
    plt.ylabel("LL")
    lw = 2
    plt.plot(clusters, sse, lw=lw)
    # plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()
