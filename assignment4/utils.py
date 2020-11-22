import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show_rewards_curve(rewards, filename, environment, env_map, alpha, decay_rate):
    plt.title("Rewards vs Episodes - Environment: {}_{}".format(env_map, environment))
    plt.xlabel(r"episodes")
    plt.ylabel(r"Rewards")
    lw = 2
    plt.plot(np.arange(rewards.shape[0]), rewards, label="learning rate: {}, decay_rate: {}".format(alpha, decay_rate),
             lw=lw)
    plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()
    plt.close()


def plot_metrics_table(data, filename):
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    df = pd.DataFrame(data=data)

    ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    fig.tight_layout()

    plt.savefig(filename, dpi=200)
    plt.close()


def show_frozen_lake_policy(policy, env_map, map_value, filename):
    size = len(map_value)
    dis_map = np.zeros((size, size))
    for i, row in enumerate(map_value):
        for j, loc in enumerate(row):
            if loc == "S":
                dis_map[i, j] = 0
            elif loc == "F":
                dis_map[i, j] = 0
            elif loc == "H":
                dis_map[i, j] = -1
            elif loc == "G":
                dis_map[i, j] = 1

    pol = np.asarray(policy).reshape((size, size))
    plt.imshow(dis_map, interpolation="nearest")
    # plt.title = "{}".format(env_map)

    for i in range(pol.shape[0]):
        for j in range(pol.shape[1]):
            arrow = '\u2190'
            if pol[i, j] == 1:
                arrow = '\u2193'
            elif pol[i, j] == 2:
                arrow = '\u2192'
            elif pol[i, j] == 3:
                arrow = '\u2191'
            plt.text(j, i, arrow,
                            ha="center", va="center", color="w")
    plt.savefig(filename)
    plt.close()
