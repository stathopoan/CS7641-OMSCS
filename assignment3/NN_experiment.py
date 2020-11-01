import time

from sklearn.metrics import accuracy_score, roc_curve, auc, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
# from utils import show_validation_curve, show_learning_curve, show_ROC, show_confusion_marix


def run_NN(X, y, alg, dataset, algorithm=None):
    print("...NN LEARNER....")
    if alg is None:
        DR_FLAG = "NoDR"
    else:
        DR_FLAG = "YesDR"

    if algorithm is None:
        algorithm="No_clustering"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9098, shuffle=True)

    clfnn = MLPClassifier(solver='sgd', activation='tanh', warm_start=True, hidden_layer_sizes=(10, 5, 2), random_state=1, max_iter=3000)

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

    if alg is None:
        show_loss_curve(best_model, grid_results.best_params_['learning_rate_init'], dataset,
                        "plots\\{}\\{}\\{}_NN_loss_curve.png".format(dataset, DR_FLAG, algorithm))
        show_accuracy_vs_epoch_curve(clfnn, X_train, y_train, X_test, y_test, dataset,
                                     "plots\\{}\\{}\\{}_NN_accuracy_vs_epoch_curve.png".format(dataset, DR_FLAG,
                                                                                                   algorithm))
        show_learning_curve(best_model, X_train, y_train,
                            "plots\\{}\\{}\\{}_NN_learning_curve.png".format(dataset, DR_FLAG, algorithm),
                            dataset)
        show_confusion_marix(best_model, X_test, y_test, dataset,
                             "plots\\{}\\{}\\{}_NN_confusion_matrix.png".format(dataset, DR_FLAG, algorithm))
    else:
        show_loss_curve(best_model, grid_results.best_params_['learning_rate_init'], dataset, "plots\\{}\\{}\\{}\\{}_NN_loss_curve.png".format(dataset, DR_FLAG, alg, algorithm))
        show_accuracy_vs_epoch_curve(clfnn, X_train, y_train, X_test, y_test, dataset, "plots\\{}\\{}\\{}\\{}_NN_accuracy_vs_epoch_curve.png".format(dataset, DR_FLAG, alg, algorithm))
        show_learning_curve(best_model, X_train, y_train, "plots\\{}\\{}\\{}\\{}_NN_learning_curve.png".format(dataset, DR_FLAG, alg, algorithm), dataset)
        show_confusion_marix(best_model, X_test, y_test, dataset,
                             "plots\\{}\\{}\\{}\\{}_NN_confusion_matrix.png".format(dataset, DR_FLAG, alg, algorithm))

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    if alg is None:
        show_ROC(fpr, tpr, roc_auc, dataset,
                 "plots\\{}\\{}\\{}_NN_ROC.png".format(dataset, DR_FLAG, algorithm))
    else:
        show_ROC(fpr, tpr, roc_auc, dataset, "plots\\{}\\{}\\{}\\{}_NN_ROC.png".format(dataset, DR_FLAG, alg, algorithm))



    print('NN: Fitting time (train data): %f seconds' % train_time)
    print('NN: Inference time (test data): %f seconds' % test_time)
    print('NN: Accuracy: %f' % nn_accuracy)

    return train_time, test_time, nn_accuracy

def show_loss_curve(model, lr, dataset, filename):
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title("Learning rate =" + str(lr))
    plt.plot(model.loss_curve_)
    plt.savefig(filename)
    plt.close()

# Reference: https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
def show_accuracy_vs_epoch_curve(model, X_train,y_train, X_test, y_test, dataset, filename):
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
    plt.close()

def show_ROC(fpr, tpr, roc_auc, dataset, filename):
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
    plt.savefig(filename)
    plt.close()

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

def show_confusion_marix(clf, X_test, y_test, dataset, filename):
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    disp.ax_.set_title("Confusion matrix - Dataset: {}".format(dataset))
    plt.savefig(filename)
    plt.clf()