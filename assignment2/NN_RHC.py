from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import mlrose_hiive as mlrose
import time
import matplotlib.pyplot as plt
import numpy as np

from utils import plot_accuracy_curve, plot_fitness_curve, plot_time_curve


def run_RHC(X, y):
    print("...RHC LEARNER....")
    iterations = [100, 200, 500, 1000, 2000, 5000]
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
                                            algorithm='random_hill_climb', max_iters=num_iterations,
                                            bias=True, is_classifier=True, learning_rate=0.1, clip_max=5,
                                            early_stopping=False, max_attempts=100, curve=True, restarts=5,
                                            random_state=state)

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

    plot_fitness_curve(curves, "rhc", filename="plots\\NN\\rhc_loss_curve.png")
    plot_time_curve(train_times, iterations, "rhc", filename="plots\\NN\\rhc_time_curve.png")
    plot_accuracy_curve(y_test_accuracies, y_train_accuracies, iterations, "rhc", filename= "plots\\NN\\rhc_accuracy_curve.png")

    # Show different fitness scores with every run "restart"
    random_states = [0, 50, 32, 333, 87652, 726182, 365, 290]
    fitness_scores = []

    for state in random_states:
        nn_model = mlrose.NeuralNetwork(hidden_nodes=[5, 2], activation='relu',
                                        algorithm='random_hill_climb', max_iters=1000,
                                        bias=True, is_classifier=True, learning_rate=0.2, clip_max=5,
                                        early_stopping=False, max_attempts=100, curve=True, restarts=0,
                                        random_state=state)
        nn_model.fit(X_train, y_train)
        fitness_scores.append( nn_model.fitness_curve[-1])

    plot_random_restart_fitness(fitness_scores,"plots\\NN\\rhc_random_restarts.png")



    return np.mean(y_test_accuracies[:, -1], axis=0), np.mean(train_times[:,-1], axis=0)


def plot_random_restart_fitness(fitness_scores, filename):
    plt.figure()
    plt.plot(np.arange(len(fitness_scores)), fitness_scores, 'blue', label='Training time')
    # plt.plot(iterations, test_time, color='red', label='Testing time')
    plt.xlabel('Runs')
    plt.ylabel('Loss')
    plt.title("Loss vs Random Starts - rhc (1000 iterations)")
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.clf()