import time

from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from utils import show_validation_curve, show_learning_curve, show_ROC, show_confusion_marix


def run_NN(X, y, dataset):
    print("...NN LEARNER....")

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    X = scaling.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9098, shuffle=True)

    clfnn = MLPClassifier(solver='sgd', activation='tanh', warm_start=True, hidden_layer_sizes=(10, 5, 2), random_state=1, max_iter=3000)
    clfnn.fit(X_train, y_train)
    y_pred = clfnn.predict(X_test)
    nn_accuracy = accuracy_score(y_test, y_pred)

    print("accuracy of default model of n=NN: {}".format(nn_accuracy))

    alpha_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    show_validation_curve(param_range = alpha_range, estimator=clfnn, X=X_train, y=y_train, scoring="accuracy", hyper_param="alpha", filename="plots\\{}\\NN_valid_curve_alpha.png".format(dataset), dataset=dataset)

    lr_range = [0.0001, 0.001, 0.005, 0.01, 0.1, 1]
    show_validation_curve(param_range=lr_range, estimator=clfnn, X=X_train, y=y_train, scoring="accuracy",
                          hyper_param="learning_rate_init", filename="plots\\{}\\NN_valid_curve_lr.png".format(dataset),
                          dataset=dataset)

    grid_params = {'alpha': [0.001, 0.01, 0.1, 1, 5, 10], 'learning_rate_init': [0.001, 0.005 ,0.01, 0.05, 0.1],
                   'activation': ['tanh', 'relu'], 'solver': ['sgd', 'adam'], 'hidden_layer_sizes': [(10,5,2), (5,), (10,10,5), (10,), (5,2)]}

    grid_results = GridSearchCV(clfnn, grid_params, cv=5, verbose=1, n_jobs=-1)
    grid_results.fit(X_train, y_train)

    print("NN best params found: " + str(grid_results.best_params_))

    best_model = MLPClassifier(random_state=1,
                               hidden_layer_sizes=grid_results.best_params_['hidden_layer_sizes'],
                               solver=grid_results.best_params_['solver'],
                               activation=grid_results.best_params_['activation'],
                               max_iter=5000, alpha=grid_results.best_params_['alpha'],
                               learning_rate_init = grid_results.best_params_['learning_rate_init'] )
    tstart = time.time()
    best_model.fit(X_train, y_train)
    tend = time.time()
    train_time = tend - tstart
    tstart = time.time()
    y_pred = best_model.predict(X_test)
    tend = time.time()
    test_time = tend - tstart
    nn_accuracy = accuracy_score(y_test, y_pred)

    show_loss_curve(best_model, grid_results.best_params_['learning_rate_init'], "plots\\{}\\NN_loss_curve.png".format(dataset), dataset)

    show_accuracy_vs_epoch_curve(clfnn, X_train, y_train, X_test, y_test, "plots\\{}\\NN_accuracy_vs_epoch_curve.png".format(dataset), dataset)

    show_learning_curve(best_model, X_train, y_train, "plots\\{}\\NN_learning_curve.png".format(dataset), dataset)

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    show_ROC(fpr, tpr, roc_auc, "\\{}\\NN_ROC.png".format(dataset), dataset)

    show_confusion_marix(best_model, X_test, y_test, "plots\\{}\\NN_confusion_matrix.png".format(dataset), dataset)

    print('NN: Fitting time (train data): %f seconds' % train_time)
    print('NN: Inference time (test data): %f seconds' % test_time)
    print('NN: Accuracy: %f' % nn_accuracy)

    return train_time, test_time, nn_accuracy

def show_loss_curve(model, lr, filename, dataset):
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title("Learning rate =" + str(lr))
    plt.plot(model.loss_curve_)
    plt.savefig(filename)
    plt.clf()

# Reference: https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
def show_accuracy_vs_epoch_curve(model, X_train,y_train, X_test, y_test, filename, dataset):
    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 1000
    N_BATCH = 128
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        # print('epoch: ', epoch)
        # SHUFFLINg
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            model.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        scores_train.append(model.score(X_train, y_train))

        # SCORE TEST
        scores_test.append(model.score(X_test, y_test))

        epoch += 1

    """ Plot """
    plt.plot(scores_train, color="darkorange", alpha=0.2, label='Training score', lw=2)
    plt.plot(scores_test, color="navy", alpha=0.2, label='Cross-validation score', lw=2)
    plt.title("Accuracy over epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.legend(loc="best")
    # plt.show()
    plt.savefig(filename)
    plt.clf()
