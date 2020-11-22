from agents import QLearning
import numpy as np

from utils import plot_metrics_table, show_rewards_curve, show_frozen_lake_policy


def run_Q_learning_experiment(envs, MAPS):
    print("###  RUNNING FROZENLAKE ENVIRONMENT WITH QLEARNING AGENT  ###")
    lrs = [0.8, 0.01, 0.1]
    decay_rates = [0.0001, 0.001]
    for env, env_map in zip(envs, MAPS):
        data = np.zeros((len(lrs) * len(decay_rates), 5))
        i = 0
        for lr in lrs:
            for decay_rate in decay_rates:
                Q_learning_agent = QLearning(num_episodes=20000, num_iterations=200, seed=4234, env=env, decay_rate=decay_rate, alpha=lr)

                policy, mean_rewards, mean_reward, mean_iteration, time_spent = Q_learning_agent.train()
                data[i, :] = lr, decay_rate, mean_reward, time_spent, mean_iteration
                i += 1

                show_rewards_curve(rewards=mean_rewards, filename="plots\\Q\\{}_frozen_lake_Q_learning_lr_{}_decay_param_{}.png".format(env_map, Q_learning_agent.alpha, Q_learning_agent.decay_rate), environment="frozen_lake", env_map=env_map, alpha=Q_learning_agent.alpha, decay_rate=Q_learning_agent.decay_rate)
                show_frozen_lake_policy(policy=policy, env_map=env_map, map_value=MAPS[env_map], filename="plots\\Q\\{}_frozen_lake_Q_learning_lr_{}_decay_param_{}_policy_map.png".format(env_map, Q_learning_agent.alpha, Q_learning_agent.decay_rate))

        data = {'Learning Rate': data[:, 0], 'Decay Rate': data[:, 1], 'Reward': data[:, 2], 'Time spent': data[:, 3],
                'Mean Iterations': data[:, 4]}
        plot_metrics_table(data, filename="plots\\Q\\{}_frozen_lake_Q_learning_metrics_table.png".format(env_map))
