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


def plot_accuracy_curve(test_accuracy, train_accuracy, iterations, algorithm, filename):
    plt.figure()
    mean_train_accuracy = np.mean(train_accuracy,axis=0)
    mean_test_accuracy = np.mean(test_accuracy,axis=0)
    plt.plot(iterations, mean_test_accuracy, 'blue', label='Test Accuracy')
    plt.plot(iterations, mean_train_accuracy, color='red', label='Train Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title("Accuracy vs Iterations - ".format(algorithm))
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.clf()

def plot_time_curve(training_time, iterations,algorithm, filename):
    plt.figure()
    mean_training_time = np.nanmean(np.where(training_time != 0, training_time, np.nan), axis=0)
    plt.plot(iterations,  mean_training_time, 'blue', label='Training time')
    plt.xlabel('Iterations')
    plt.ylabel('Time(s)')
    plt.title("Time vs Iterations - {}".format(algorithm))
    plt.legend(loc='best')
    plt.savefig(filename)
    plt.clf()

def plot_aggregated_time_curve(training_time, algorithms, filename, problem_dsc):
    plt.title("Time vs Iterations - {}".format(problem_dsc))
    plt.xlabel('Iterations')
    plt.ylabel('Time(s)')
    lw = 2
    param_range = np.arange(training_time.shape[1])
    for i in np.arange(training_time.shape[0]):
        plt.semilogx(param_range, training_time[i, :], label=algorithms[i], lw=lw)
    plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def plot_fitness_curve(curves, algorithm, filename):
    fitness_mean = np.nanmean(np.where(curves != 0, curves, np.nan), axis=0)
    fitness_std = np.nanstd(np.where(curves != 0, curves, np.nan), axis=0)

    plt.title("Loss Curve ")
    plt.xlabel(r"Iterations")
    plt.ylabel("Loss")
    lw = 2
    param_range = np.arange(curves.shape[1])
    plt.plot(param_range, fitness_mean, label="Loss score", color="darkorange", lw=lw)
    plt.fill_between(param_range, fitness_mean - fitness_std, fitness_mean + fitness_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.legend(loc="best")
    plt.title("Loss vs Iterations - {}".format(algorithm))
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def plot_fitness_evaluation_curve(curves, evaluations, algorithm, filename):
    fitness_mean = np.nanmean(np.where(curves != 0, curves, np.nan), axis=0)
    # fitness_std = np.nanstd(np.where(curves != 0, curves, np.nan), axis=0)

    evaluation_mean = np.nanmean(np.where(evaluations != 0, evaluations, np.nan), axis=0)
    # evaluation_std = np.nanstd(np.where(evaluations != 0, evaluations, np.nan), axis=0)

    plt.xlabel(r"Evaluations")
    plt.ylabel("Fitness")
    lw = 2
    plt.plot(evaluation_mean, fitness_mean, label="Loss score", color="darkorange", lw=lw)
    plt.legend(loc="best")
    plt.title("Fitness vs Evaluations - {}".format(algorithm))
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def plot_aggregated_fitness_evaluation_curve(curves, evaluations, algorithms, filename, problem_dsc):
    plt.title("Fitness vs Evaluations - {}".format(problem_dsc))
    plt.xlabel(r"Evaluations")
    plt.ylabel("Fitness")
    lw = 2
    for i in np.arange(curves.shape[0]):
        plt.semilogx(evaluations[i,:], curves[i,:], label=algorithms[i], lw=lw)
    plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def plot_objective_curve(curves, algorithm, filename):
    fitness_mean = np.nanmean(np.where(curves != 0, curves, np.nan), axis=0)
    fitness_std = np.nanstd(np.where(curves != 0, curves, np.nan), axis=0)

    plt.title("Fitness vs Iterations - {}".format(algorithm))

    plt.xlabel(r"Iterations")
    plt.ylabel("Fitness")
    lw = 2
    param_range = np.arange(curves.shape[1])
    plt.plot(param_range, fitness_mean, label="Loss score", color="darkorange", lw=lw)
    plt.fill_between(param_range, fitness_mean - fitness_std, fitness_mean + fitness_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def plot_aggregated_objective_curve(curves, algorithms, filename, problem_dsc):
    plt.title("Fitness vs Iterations - problem: {}".format(problem_dsc))
    plt.xlabel(r"Iterations")
    plt.ylabel("Fitness")
    lw = 2
    param_range = np.arange(curves.shape[1])
    for i in np.arange(curves.shape[0]):
        plt.semilogx(param_range, curves[i,:], label=algorithms[i], lw=lw)
    plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()


def plot_fitness_time_curve(curves, times, algorithm, filename):
    times_mean = np.nanmean(np.where(times != 0, times, np.nan), axis=0)
    times_std = np.nanstd(np.where(times != 0, times, np.nan), axis=0)

    fitness_mean = np.nanmean(np.where(curves != 0, curves, np.nan), axis=0)
    fitness_std = np.nanstd(np.where(curves != 0, curves, np.nan), axis=0)


    plt.title("Fitting time Curve ")
    plt.xlabel(r"Time")
    plt.ylabel("Fitness")
    lw = 2
    plt.plot(times_mean, fitness_mean, label="Loss score", color="darkorange", lw=lw)
    plt.legend(loc="best")
    plt.title("Fitness vs Time - {}".format(algorithm))
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def plot_aggregated_fitness_time_curve(curves, times, algorithms, filename, problem_dsc):
    plt.title("Fitness vs Time - problem: {}".format(problem_dsc))
    plt.xlabel(r"Time(s)")
    plt.ylabel("Fitness")
    lw = 2
    for i in np.arange(curves.shape[0]):
        plt.semilogx(times[i,:], curves[i,:], label=algorithms[i], lw=lw)
    plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()

def show_metrics(fitting_time, accuracy):
    show_metric(fitting_time, "fitting time", 'seconds', "plots\\NN\\all_fitting_time.png")
    show_metric(accuracy, "Accuracy", 'score', "plots\\NN\\all_accuracy.png")



def show_metric(metric_performance, title, units, filename):
    optimizers = ['GD', 'RHC', 'SA', 'GA']
    y_pos = np.arange(len(optimizers))

    f, ax = plt.subplots(figsize=(9, 5))
    plt.barh(y_pos, metric_performance, align='center', alpha=0.5)
    plt.yticks(y_pos, optimizers)
    plt.xlabel(units)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()
