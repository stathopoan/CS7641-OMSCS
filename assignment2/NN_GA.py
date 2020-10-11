from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import time
import mlrose_hiive as mlrose

from utils import plot_fitness_curve, plot_time_curve, plot_accuracy_curve


def run_GA(X, y):
    print("...GA LEARNER....")
    iterations = [100, 200, 300, 500, 1000, 2000]
    random_states = [3, 98, 654]

    # scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    # X = scaling.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9098, shuffle=True)

    train_times = np.zeros((len(random_states), len(iterations)))
    test_times = np.zeros((len(random_states), len(iterations)))
    y_train_accuracies = np.zeros((len(random_states), len(iterations)))
    y_test_accuracies = np.zeros((len(random_states), len(iterations)))

    curves = np.zeros((len(iterations) * len(random_states), max(iterations)))

    for j, state in enumerate(random_states):
        # print (state)

        for i, num_iterations in enumerate(iterations):
            nn_model = mlrose.NeuralNetwork(hidden_nodes=[5, 2], activation='relu',
                                            algorithm='genetic_alg', max_iters=num_iterations,
                                            bias=True, is_classifier=True, learning_rate=0.01, clip_max=5,
                                            early_stopping=False, max_attempts=100, curve=True, pop_size=100,
                                            mutation_prob=0.1, random_state=state)

            tstart = time.time()
            nn_model.fit(X_train, y_train)
            tend = time.time()
            train_time = tend - tstart
            train_times[j, i] = train_time
            tstart = time.time()

            tend = time.time()
            test_time = tend - tstart
            test_times[j, i] = test_time
            y_train_accuracy = accuracy_score(y_train, nn_model.predict(X_train))
            y_train_accuracies[j, i] = y_train_accuracy
            y_test_accuracy = accuracy_score(y_test, nn_model.predict(X_test))
            y_test_accuracies[j, i] = y_test_accuracy

            curves[j * len(iterations) + i, :len(nn_model.fitness_curve)] = nn_model.fitness_curve

    plot_fitness_curve(curves, "ga", filename="plots\\NN\\ga_loss_curve.png")
    plot_time_curve(train_times, iterations, "ga", filename="plots\\NN\\ga_time_curve.png")
    plot_accuracy_curve(y_test_accuracies, y_train_accuracies, iterations, "sa",
                        filename="plots\\NN\\ga_accuracy_curve.png")

    return np.mean(y_test_accuracies[:, -1], axis=0), np.mean(train_times[:, -1], axis=0)