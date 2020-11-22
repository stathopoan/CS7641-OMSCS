from agents import ValueIteration, PolicyIteration
from utils import show_frozen_lake_policy, plot_metrics_table
import matplotlib.pyplot as plt
import numpy as np


def run_vi_pi_experiment(envs, MAPS):
    gammas = [0.5, 0.6, 0.7, 0.8, 0.9]
    theta = 0.0001
    for env, env_map in zip(envs, MAPS):
        data_vi = np.zeros((len(gammas), 6))
        data_pi = np.zeros((len(gammas), 6))

        for i, gamma in enumerate(gammas):
            # Value Iteration
            vi_agent = ValueIteration(env, max_iterations=500, discount_factor=gamma, theta=theta)
            V, pi, train_iterations, time_spent = vi_agent.train()
            show_frozen_lake_policy(policy=pi, env_map=env_map, map_value=MAPS[env_map],
                                    filename="plots\\VI\\{}_frozen_lake_VI_learning_discount_factor_{}_policy_map.png".format(
                                        env_map, vi_agent.gamma))

            rewards, episode_iterations = vi_agent.test()
            data_vi[i, :] = np.mean(rewards), np.mean(episode_iterations), train_iterations, time_spent, gamma, theta

            # Policy Iteration
            pi_agent = PolicyIteration(env, max_iterations=500, discount_factor=gamma, theta=theta)
            V, pi, train_iterations, time_spent = pi_agent.train()
            show_frozen_lake_policy(policy=pi, env_map=env_map, map_value=MAPS[env_map],
                                    filename="plots\\PI\\{}_frozen_lake_PI_learning_discount_factor_{}_policy_map.png".format(
                                        env_map, pi_agent.gamma))

            rewards, episode_iterations = pi_agent.test()
            data_pi[i, :] = np.mean(rewards), np.mean(episode_iterations), train_iterations, time_spent, gamma, theta

        show_rewards_gamma_curve(data_vi, gammas,
                           filename="plots\\VI\\{}_frozen_lake_VI_learning_rewards_vs_gamma.png".format(
                               env_map), environment="frozen_lake", env_map=env_map, theta=theta)

        show_rewards_gamma_curve(data_pi, gammas,
                                 filename="plots\\PI\\{}_frozen_lake_PI_learning_rewards_vs_gamma.png".format(
                                     env_map), environment="frozen_lake", env_map=env_map, theta=theta)

        show_iter_gamma_curve(data_vi,gammas,
                            filename="plots\\VI\\{}_frozen_lake_VI_learning_iter_per_episode_vs_gamma.png".format(
                               env_map), environment="frozen_lake", env_map=env_map, theta=theta)

        show_iter_gamma_curve(data_pi, gammas,
                              filename="plots\\PI\\{}_frozen_lake_PI_learning_iter_per_episode_vs_gamma.png".format(
                                  env_map), environment="frozen_lake", env_map=env_map, theta=theta)

        show_train_iter_gamma_curve(data_vi,gammas,
                            filename="plots\\VI\\{}_frozen_lake_VI_learning_iter_train_vs_gamma.png".format(
                               env_map), environment="frozen_lake", env_map=env_map, theta=theta)

        show_train_iter_gamma_curve(data_pi, gammas,
                                    filename="plots\\PI\\{}_frozen_lake_PI_learning_iter_train_vs_gamma.png".format(
                                        env_map), environment="frozen_lake", env_map=env_map, theta=theta)

        show_time_gamma_curve(data_vi,gammas,
                            filename="plots\\VI\\{}_frozen_lake_VI_time_train_vs_gamma.png".format(
                               env_map), environment="frozen_lake", env_map=env_map, theta=theta)

        show_time_gamma_curve(data_pi, gammas,
                              filename="plots\\PI\\{}_frozen_lake_PI_time_train_vs_gamma.png".format(
                                  env_map), environment="frozen_lake", env_map=env_map, theta=theta)

        data_VI = {'gamma': data_vi[:, 4], 'theta': data_vi[:, 5],  'Mean reward': data_vi[:, 0], 'Mean iteration/episode': data_vi[:, 1], 'Iterations(training)': data_vi[:, 2], 'Time spent': data_vi[:, 3]}
        plot_metrics_table(data_VI, filename="plots\\VI\\{}_frozen_lake_VI_metrics_table.png".format(env_map))

        data_PI = {'gamma': data_pi[:, 4], 'theta': data_pi[:, 5], 'Mean reward': data_pi[:, 0], 'Mean iteration/episode': data_pi[:, 1], 'Iterations(training)': data_pi[:, 2], 'Time spent': data_pi[:, 3]}
        plot_metrics_table(data_PI, filename="plots\\PI\\{}_frozen_lake_PI_metrics_table.png".format(env_map))

def show_rewards_gamma_curve(data, gammas, filename, environment, env_map, theta):
    plt.title("VI - Rewards vs Discount factor - Environment: {}_{}".format(env_map, environment))
    plt.xlabel(r"discount factor")
    plt.ylabel(r"Mean Reward")
    lw = 2

    mean_rewards = data[:,0]
    plt.plot(gammas, mean_rewards, label="theta: {}".format(theta),
             lw=lw)
    plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()
    plt.close()

def show_iter_gamma_curve(data, gammas, filename, environment, env_map, theta):
    plt.title("VI - Mean Iterations per episode vs Discount factor - Environment: {}_{}".format(env_map, environment))
    plt.xlabel("discount factor")
    plt.ylabel("Mean Iterations/episode (testing)")
    lw = 2

    mean_episode_iterations = data[:,1]
    plt.plot(gammas, mean_episode_iterations, label="theta: {}".format(theta),
             lw=lw)
    plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()
    plt.close()

def show_train_iter_gamma_curve(data, gammas, filename, environment, env_map, theta):
    plt.title("VI - Iterations (training) vs Discount factor - Environment: {}_{}".format(env_map, environment))
    plt.xlabel("discount factor")
    plt.ylabel("Iterations (training)")
    lw = 2

    mean_episode_iterations = data[:,2]
    plt.plot(gammas, mean_episode_iterations, label="theta: {}".format(theta),
             lw=lw)
    plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()
    plt.close()

def show_time_gamma_curve(data, gammas, filename, environment, env_map, theta):
    plt.title("VI - Time (training) vs Discount factor - Environment: {}_{}".format(env_map, environment))
    plt.xlabel("discount factor")
    plt.ylabel("Time (training)")
    lw = 2

    mean_episode_iterations = data[:,3]
    plt.plot(gammas, mean_episode_iterations, label="theta: {}".format(theta),
             lw=lw)
    plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()
    plt.close()
