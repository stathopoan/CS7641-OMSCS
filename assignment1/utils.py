from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import validation_curve, learning_curve


def load_white_wine_data():
    input = './data/winequality-white.csv'
    df = pd.read_csv(input, header=0, sep=';')
    # dataset = np.loadtxt(input, delimiter=';')
    return df

def load_heart_disease_data():
    input = './data/reprocessed.hungarian.data'
    names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(input, sep=" ", header=None, names=names)
    return df

def show_ROC(fpr, tpr, roc_auc, figname, dataset):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(' Area under ROC - Dataset: {}'.format(dataset))
    plt.legend(loc="lower right")
    plt.savefig("plots\{}".format(figname))
    plt.clf()


# Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto
# -examples-model-selection-plot-validation-curve-py
def show_validation_curve(param_range, estimator, X, y, scoring, hyper_param, filename, dataset):
    # print ("Start validation curve")
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=hyper_param, param_range=param_range,
                                                 scoring=scoring, n_jobs=8)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve - Dataset: {}".format(dataset))
    plt.xlabel(r"{}".format(hyper_param))
    plt.ylabel(r"Score - {}".format(scoring))
    # plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()
    # print("End validation curve")
    return test_scores_mean

# Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def show_learning_curve(clf, X_train, y_train, filename, dataset, train_sizes=None):
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)

    # Plot learning curve
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(clf, X_train, y_train,
                                                                          train_sizes=train_sizes, cv=5,
                                                                          n_jobs=4, shuffle=True, random_state=45,
                                                                          return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    _, axes = plt.subplots(1, 1, figsize=(10, 5))

    plt.title("Learning Curves - Dataset: {}".format(dataset))
    plt.xlabel(r"Training examples")
    plt.ylabel(r"Cross validation score")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")

    plt.savefig(filename)
    plt.clf()

def show_confusion_marix(clf, X_test, y_test, filename, dataset ):
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    disp.ax_.set_title("Confusion matrix - Dataset: {}".format(dataset))
    plt.savefig(filename)
    plt.clf()

def show_metrics(fitting_time_wine, inference_time_wine, accuracy_wine, fitting_time_heart, inference_time_heart, accuracy_heart):
    show_metric(fitting_time_wine, "fitting time - Dataset: wine", 'seconds', "plots\\wine\\fitting_time.png")
    show_metric(inference_time_wine, "Inference time - Dataset: wine", 'seconds', "plots\\wine\\inference_time.png")
    show_metric(accuracy_wine, "Accuracy - Dataset: wine", 'score', "plots\\wine\\accuracy.png")

    show_metric(fitting_time_heart, "fitting time - Dataset: heart disease", 'seconds', "plots\\heart\\fitting_time.png")
    show_metric(inference_time_heart, "Inference time - Dataset: heart disease", 'seconds', "plots\\heart\\inference_time.png")
    show_metric(accuracy_heart, "Accuracy - Dataset: heart disease", 'score', "plots\\heart\\accuracy.png")



def show_metric(metric_performance, title, units, filename):
    classifiers = ['Decision Tree', 'Adaboost', 'SVM', 'KNN', 'NeuralNetwork']
    y_pos = np.arange(len(classifiers))

    f, ax = plt.subplots(figsize=(9, 5))
    plt.barh(y_pos, metric_performance, align='center', alpha=0.5)
    plt.yticks(y_pos, classifiers)
    plt.xlabel(units)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()