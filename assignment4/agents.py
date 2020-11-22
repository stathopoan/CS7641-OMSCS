import time
from collections import deque
import gym

import numpy as np
import sys


class QLearning:
    def __init__(self, num_episodes=1000, num_iterations=500, seed=7782, env=gym.make("FrozenLake-v0"),
                 decay_rate=0.005, alpha=0.8):
        self.num_episodes = num_episodes
        self.num_iterations = num_iterations
        self.seed = seed
        self.env = env
        self.alpha = alpha
        self.gamma = 0.9
        self.rewards = np.zeros(self.num_episodes)
        self.rewards_queue = deque(maxlen=100)
        self.mean_rewards = np.zeros(self.num_episodes)
        self.iterations = np.zeros(self.num_episodes)
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = decay_rate

    def train(self):
        np.random.seed(self.seed)
        Q = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        epsilon = self.max_epsilon
        alpha = self.alpha
        start = time.time()
        for episode in range(self.num_episodes):
            episode_reward = 0
            if (episode + 1) % 100 == 0:
                print("\rEpisode {}/{}.".format(episode + 1, self.num_episodes), end="")
                sys.stdout.flush()

            state = self.env.reset()
            for iteration in range(self.num_iterations):
                # env.render()
                # print(    state)
                action = self.choose_action(Q, state, epsilon)
                state_new, reward, done, info = self.env.step(action)
                Q[state, action] = Q[state, action] + alpha * (
                        reward + self.gamma * np.max(Q[state_new, :]) - Q[state, action])
                episode_reward += reward

                state = state_new

                epsilon = max(self.max_epsilon - self.decay_rate * episode, self.min_epsilon)

                if done or iteration == self.num_iterations - 1:
                    self.rewards[episode] = episode_reward
                    self.rewards_queue.append(episode_reward)
                    self.mean_rewards[episode] = np.mean(self.rewards_queue)
                    self.iterations[episode] = iteration
                    break

        self.env.close()
        end = time.time()
        time_spent = end - start
        return np.argmax(Q, axis=1), self.mean_rewards, np.mean(self.rewards), np.mean(self.iterations), time_spent

    def choose_action(self, Q, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.env.action_space.n)
        else:
            return np.argmax(Q[state, :])


# Reference: https://towardsdatascience.com/value-iteration-to-solve-openai-gyms-frozenlake-6c5e7bf0a64d
def argmax(env, V, pi, s, gamma):
    qe = np.zeros(env.action_space.n)
    for a in range(env.action_space.n):
        P = np.array(env.P[s][a])
        q = 0
        for i in range(P.shape[0]):
            s_ = int(P[i][1])
            p = P[i][0]
            r = P[i][2]
            q += p * (r + gamma * V[s_])
            qe[a] = q
    best_action = np.argmax(qe)
    pi[s] = best_action
    return best_action


# Reference: https://towardsdatascience.com/value-iteration-to-solve-openai-gyms-frozenlake-6c5e7bf0a64d
class ValueIteration:
    def __init__(self, env, max_iterations=500, discount_factor=0.9, theta=0.0001):
        self.env = env
        self.max_iterations = max_iterations
        self.gamma = discount_factor
        self.theta = theta
        self.V = np.zeros(self.env.observation_space.n)
        self.optimal_policy = np.zeros(self.env.observation_space.n)

    def train(self):
        start_time = time.time()
        V = np.zeros(self.env.observation_space.n)
        iterations = 0
        while True:
            iterations += 1
            delta = 0
            for s in range(self.env.observation_space.n):
                v = V[s]
                self.bellman_optimality_update(self.env, V, s, self.gamma)
                delta = max(delta, abs(v - V[s]))
            if delta < self.theta:
                break
        optimal_policy = np.zeros(self.env.observation_space.n)
        for s in range(self.env.observation_space.n):
            argmax(self.env, V, optimal_policy, s, self.gamma)

        self.V = V
        self.optimal_policy = optimal_policy
        end_time = time.time()
        time_spent = end_time - start_time
        return V, optimal_policy, iterations, time_spent

    def bellman_optimality_update(self, env, V, s, gamma):
        optimal_policy = np.zeros(self.env.observation_space.n)
        # Find action which gives maximum value
        greedy_action = argmax(env, V, optimal_policy, s, gamma)
        # Take greedy action and update value
        P = np.array(env.P[s][greedy_action])
        v = 0
        for i in range(P.shape[0]):
            s_ = int(P[i][1])
            p = P[i][0]
            r = P[i][2]

            v += p * (r + gamma * V[s_])

        V[s] = v
        return V[s]

    def test(self, episodes=1000):
        rewards = np.zeros(episodes)
        iterations = np.zeros(episodes)

        for episode in range(episodes):
            episode_reward = 0
            if (episode + 1) % 100 == 0:
                print("\rEpisode {}/{}.".format(episode + 1, episodes), end="")
                sys.stdout.flush()
            state = self.env.reset()
            iteration = 0
            done = False
            while not done and iteration < self.max_iterations:
                state_new, reward, done, info = self.env.step(self.optimal_policy[state])
                episode_reward += reward
                state = state_new
                iteration += 1

            rewards[episode] = episode_reward
            iterations[episode] = iteration
        self.env.close()

        return rewards, iterations


class PolicyIteration:
    def __init__(self, env, max_iterations=500, discount_factor=0.9, theta=0.0001):
        self.env = env
        self.max_iterations = max_iterations
        self.gamma = discount_factor
        self.theta = theta
        self.V = np.zeros(self.env.observation_space.n)
        self.optimal_policy = np.zeros(self.env.observation_space.n)

    def train(self):
        start_time = time.time()
        V = np.zeros(self.env.observation_space.n)
        optimal_policy = np.zeros(self.env.observation_space.n)
        iterations = 0
        while True:
            iterations += 1
            # Policy Evaluation
            while True:
                delta = 0
                for s in range(self.env.observation_space.n):
                    v = V[s]
                    self.update_V(self.env, V, s, optimal_policy[s], self.gamma)
                    delta = max(delta, abs(v - V[s]))
                if delta < self.theta:
                    break

            # Policy Improvement
            policy_stable = True
            for s in range(self.env.observation_space.n):
                old_action = optimal_policy[s]
                argmax(self.env, V, optimal_policy, s, self.gamma)
                if old_action != optimal_policy[s]:
                    policy_stable = False
            if policy_stable:
                break

        self.V = V
        self.optimal_policy = optimal_policy
        end_time = time.time()
        time_spent = end_time - start_time
        return V, optimal_policy, iterations, time_spent

    def test(self, episodes=1000):
        rewards = np.zeros(episodes)
        iterations = np.zeros(episodes)

        for episode in range(episodes):
            episode_reward = 0
            if (episode + 1) % 100 == 0:
                print("\rEpisode {}/{}.".format(episode + 1, episodes), end="")
                sys.stdout.flush()
            state = self.env.reset()
            iteration = 0
            done = False
            while not done and iteration < self.max_iterations:
                state_new, reward, done, info = self.env.step(self.optimal_policy[state])
                episode_reward += reward
                state = state_new
                iteration += 1

            rewards[episode] = episode_reward
            iterations[episode] = iteration
        self.env.close()

        return rewards, iterations

    def update_V(self, env, V, s, a, gamma):
        P = np.array(env.P[s][a])
        v = 0
        for i in range(P.shape[0]):
            s_ = int(P[i][1])
            p = P[i][0]
            r = P[i][2]

            v += p * (r + gamma * V[s_])
        V[s] = v
