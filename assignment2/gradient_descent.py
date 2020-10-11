import time

from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# From my assignment 1
def run_NN(X, y):
    print("...NN LEARNER....")

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    X = scaling.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9098, shuffle=True)

    best_model = MLPClassifier(solver='sgd', activation='relu',hidden_layer_sizes=(5, 2), random_state=1, learning_rate_init =0.005, alpha=0.001, max_iter=3000)

    tstart = time.time()
    best_model.fit(X_train, y_train)
    tend = time.time()
    train_time = tend - tstart
    tstart = time.time()
    y_pred = best_model.predict(X_test)
    tend = time.time()
    test_time = tend - tstart
    nn_accuracy = accuracy_score(y_test, y_pred)

    show_loss_curve(best_model, 0.005, "GD", "plots\\NN\\gradient_descent_loss_curve.png")

    print('NN: Fitting time (train data): %f seconds' % train_time)
    print('NN: Inference time (test data): %f seconds' % test_time)
    print('NN: Accuracy: %f' % nn_accuracy)

    return nn_accuracy, train_time

def show_loss_curve(model, lr, algorithm, filename):
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title("Learning rate =" + str(lr))
    plt.plot(model.loss_curve_)
    plt.title("Loss vs epochs - {}".format(algorithm))
    plt.savefig(filename)
    plt.clf()
