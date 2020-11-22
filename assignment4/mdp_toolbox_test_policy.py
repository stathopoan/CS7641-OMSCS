import sys
from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
import numpy as np

from utils import plot_metrics_table


class PolicyIterationMDP:
    def __init__(self, P, R, max_iterations=1000, gamma=0.9):
        self.P = P
        self.R = R
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.pi = PolicyIteration(transitions=self.P, reward=self.R, gamma=self.gamma, max_iter=self.max_iterations)

    def train(self):
        self.pi.run()
        return self.pi.V, self.pi.policy, self.pi.iter, self.pi.time

    def test(self, iterations=1000):
        num_states = self.P.shape[-1]
        rewards = np.zeros(num_states)

        for state in range(num_states):
            state_reward = 0
            for iteration in range(iterations):
                current_state = state
                iteration_reward = 0
                disc_rate = 1
                while True:
                    action = self.pi.policy[current_state]
                    p = self.P[action][current_state]
                    candidate_states = np.nonzero(p)[0]
                    next_state = np.random.choice(candidate_states, 1, p=p[candidate_states])[0]
                    reward = self.R[current_state][action]
                    iteration_reward += reward
                    disc_rate *= self.gamma
                    if next_state == 0:
                        break
                    current_state = next_state

                state_reward += iteration_reward
            rewards[state] = state_reward / iterations

        mean_reward = np.mean(rewards)
        return mean_reward


class ValueIterationMDP:
    def __init__(self, P, R, max_iterations=500, epsilon=0.0001, gamma=0.9):
        self.P = P
        self.R = R
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.gamma = gamma
        self.vi = ValueIteration(transitions=self.P, reward=self.R, gamma=self.gamma, epsilon=self.epsilon,
                                 max_iter=self.max_iterations)

    def train(self):
        self.vi.run()
        return self.vi.V, self.vi.policy, self.vi.iter, self.vi.time

    def test(self, iterations=1000):
        num_states = self.P.shape[-1]
        rewards = np.zeros(num_states)

        for state in range(num_states):
            state_reward = 0
            for iteration in range(iterations):
                current_state = state
                iteration_reward = 0
                disc_rate = 1
                while True:
                    action = self.vi.policy[current_state]
                    p = self.P[action][current_state]
                    candidate_states = np.nonzero(p)[0]
                    next_state = np.random.choice(candidate_states, 1, p=p[candidate_states])[0]
                    reward = self.R[current_state][action]
                    iteration_reward += reward
                    disc_rate *= self.gamma
                    if next_state == 0:
                        break
                    current_state = next_state

                state_reward += iteration_reward
            rewards[state] = state_reward / iterations

        mean_reward = np.mean(rewards)
        return mean_reward

class QLearningMDP:
    def __init__(self, P, R, iterations=10000, epsilon=0.0001, epsilon_decay=0.99, epsilon_min=0.1, gamma=0.9, alpha=0.1, alpha_min=0.001, alpha_decay = 0.99):
        self.P = P
        self.R = R
        self.n_iterations = iterations
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.alpha = alpha
        self.epsilon_min = epsilon_min
        self.Q = QLearning(self.P, self.R, gamma=self.gamma, alpha_decay=self.alpha_decay, alpha_min=self.alpha_min, epsilon=self.epsilon,
                  epsilon_decay=self.epsilon_decay, epsilon_min=self.epsilon_min, n_iter=self.n_iterations, alpha=self.alpha)

    def train(self):
        self.Q.run()
        return self.Q.V, self.Q.policy, self.Q.time

    def test(self, iterations=1000):
        num_states = self.P.shape[-1]
        rewards = np.zeros(num_states)

        for state in range(num_states):
            state_reward = 0
            for iteration in range(iterations):
                current_state = state
                iteration_reward = 0
                disc_rate = 1
                while True:
                    action = self.Q.policy[current_state]
                    p = self.P[action][current_state]
                    candidate_states = np.nonzero(p)[0]
                    next_state = np.random.choice(candidate_states, 1, p=p[candidate_states])[0]
                    reward = self.R[current_state][action]
                    iteration_reward += reward
                    disc_rate *= self.gamma
                    if next_state == 0:
                        break
                    current_state = next_state

                state_reward += iteration_reward
            rewards[state] = state_reward / iterations

        mean_reward = np.mean(rewards)
        return mean_reward


def run_vi_pi_forest_experiment(P, R, no_states):
    print("###  RUNNING MDP FOREST FOR S={} ###".format(no_states))
    # gammas = [0.5, 0.6, 0.7, 0.8, 0.9]
    gamma = 0.9
    epsilons = [0.1, 0.01, 0.001]
    data_vi = np.zeros((len(epsilons), 5))
    i = 0
    print("Value Iteration.........")
    for epsilon in epsilons:
        print("Running VI for gamma: {}, epsilon: {}".format(gamma, epsilon))
        vi_agent = ValueIterationMDP(P, R, gamma=gamma, epsilon=epsilon)
        V, pi, train_iterations, time_spent = vi_agent.train()
        print("Best policy found: {}".format(pi))
        mean_reward = vi_agent.test()
        data_vi[i, :] = epsilon, gamma, mean_reward, train_iterations, time_spent
        i += 1

    data_VI = {'epsilon': data_vi[:, 0], 'gamma': data_vi[:, 1], 'Mean reward': data_vi[:, 2],
               'Iterations': data_vi[:, 3], 'Time spent': data_vi[:, 4]}
    plot_metrics_table(data_VI, filename="plots\\VI\\{}_forest_VI_metrics_table.png".format(no_states))

    i=0
    data_pi = np.zeros((1, 4))
    print("Policy Iteration.........")
    # for gamma in gammas:
    print("Running PI for gamma: {}".format(gamma))
    pi_agent = PolicyIterationMDP(P, R, gamma=gamma)
    V, pi, train_iterations, time_spent = pi_agent.train()
    print("Best policy found: {}".format(pi))
    mean_reward = pi_agent.test()
    data_pi[i, :] = gamma, mean_reward, train_iterations, time_spent
    # i += 1

    data_PI = {'gamma': data_pi[:, 0], 'Mean reward': data_pi[:, 1],
               'Iterations': data_pi[:, 2], 'Time spent': data_pi[:, 3]}
    plot_metrics_table(data_PI, filename="plots\\PI\\{}_forest_PI_metrics_table.png".format(no_states))

def run_q_learning_forest_experiment(P, R, no_states):
    print("###  RUNNING MDP FOREST FOR S={} ###".format(no_states))
    iterations = [10000, 20000]
    epsilons = [0.01, 0.001]
    alpha_decs = [0.99, 0.999]
    data_q = np.zeros((len(iterations) * len(epsilons)*len(alpha_decs), 7))
    i = 0
    print("Q Learning.........")
    for iteration in iterations:
        for epsilon in epsilons:
            for alpha_dec in alpha_decs:
                print("Running Q learning for iterations: {}, epsilon: {}, alpha_dec: {}".format(iteration, epsilon, alpha_dec))
                q_agent = QLearningMDP(P, R, iterations=iteration ,epsilon = epsilon, alpha_decay=alpha_dec)
                V, pi, time_spent = q_agent.train()
                print("Best policy found: {}".format(pi))
                mean_reward = q_agent.test()
                data_q[i, :] = q_agent.gamma, q_agent.epsilon, q_agent.alpha_decay, q_agent.alpha,  q_agent.n_iterations, mean_reward, time_spent
                i += 1

    data_PI = {'gamma': data_q[:, 0],'epsilon':data_q[:,1], 'alpha_decay':data_q[:,2], 'alpha':data_q[:,3], 'Iterations': data_q[:, 4],
               'Mean reward': data_q[:, 5], 'Time spent': data_q[:, 6]}
    plot_metrics_table(data_PI, filename="plots\\Q\\{}_forest_QLearning_metrics_table.png".format(no_states))