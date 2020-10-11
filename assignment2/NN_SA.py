from mlrose_hiive import ExpDecay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import mlrose_hiive as mlrose
import time
import matplotlib.pyplot as plt

from utils import plot_fitness_curve, plot_time_curve, plot_accuracy_curve


def run_SA(X, y):
    print("...SA LEARNER....")
    iterations = [100, 200, 500, 1000, 2000, 5000]
    random_states = [3, 98, 654]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9098, shuffle=True)

    train_times = np.zeros((len(random_states), len(iterations)))
    test_times = np.zeros((len(random_states), len(iterations)))
    y_train_accuracies = np.zeros((len(random_states), len(iterations)))
    y_test_accuracies = np.zeros((len(random_states), len(iterations)))

    curves = np.zeros((len(iterations)*len(random_states), max(iterations)))

    exp_decay = ExpDecay(init_temp=100,
                         exp_const=0.1,
                         min_temp=0.001)

    for j, state in enumerate(random_states):
        # print (state)

        for i, num_iterations in enumerate(iterations):
            nn_model = mlrose.NeuralNetwork(hidden_nodes=[5, 2], activation='relu',
                                            algorithm='simulated_annealing', max_iters=num_iterations,
                                            bias=True, is_classifier=True, learning_rate=0.1, clip_max=5,
                                            early_stopping=False, max_attempts=100, curve=True, schedule=exp_decay,
                                            random_state=state)

            tstart = time.time()
            nn_model.fit(X_train, y_train)
            tend = time.time()
            train_time = tend - tstart
            train_times[j,i] = train_time
            tstart = time.time()

            tend = time.time()
            test_time = tend - tstart
            test_times[j, i] = test_time
            y_train_accuracy = accuracy_score(y_train, nn_model.predict(X_train))
            y_train_accuracies[j, i] = y_train_accuracy
            y_test_accuracy = accuracy_score(y_test, nn_model.predict(X_test))
            y_test_accuracies[j,i] = y_test_accuracy


            curves[j*len(iterations)+i, :len(nn_model.fitness_curve)] = nn_model.fitness_curve

    plot_fitness_curve(curves, "sa", filename="plots\\NN\\sa_loss_curve.png")
    plot_time_curve(train_times, iterations, "sa", filename="plots\\NN\\sa_time_curve.png")
    plot_accuracy_curve(y_test_accuracies, y_train_accuracies, iterations, "sa",
                        filename="plots\\NN\\sa_accuracy_curve.png")

    # Show different fitness scores for different starting Temperatures
    starting_temperatures = [1, 10, 20, 40, 50, 80, 100, 1000]
    fitness_scores = []

    for init_temp in starting_temperatures:
        exp_decay = ExpDecay(init_temp=init_temp,
                             exp_const=0.1,
                             min_temp=0.001)
        nn_model = mlrose.NeuralNetwork(hidden_nodes=[5, 2], activation='relu',
                                        algorithm='simulated_annealing', max_iters=1000,
                                        bias=True, is_classifier=True, learning_rate=1.0, clip_max=5,
                                        early_stopping=False, max_attempts=100, curve=True, schedule=exp_decay,
                                        random_state=3)
        nn_model.fit(X_train, y_train)
        fitness_scores.append(nn_model.fitness_curve[-1])

    plot_different_temperatures_fitness(fitness_scores, starting_temperatures,  "plots\\NN\\sa_initial_temp_curve.png")


    return np.mean(y_test_accuracies[:, -1], axis=0), np.mean(train_times[:,-1], axis=0)

def plot_different_temperatures_fitness(fitness_scores, starting_temperatures, filename):
    plt.figure()
    plt.plot(starting_temperatures, fitness_scores, 'blue', label='Training time')
    # plt.plot(iterations, test_time, color='red', label='Testing time')
    plt.xlabel('Initial Temperature')
    plt.ylabel('Loss')
    plt.title("Loss vs Initial Temperature - sa (1000 iterations)")
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.clf()