import time

from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import show_validation_curve, show_ROC, show_learning_curve, show_confusion_marix


def run_KNN(X, y, dataset):
    print("...KNN LEARNER....")

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    X = scaling.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=9098, shuffle=True)

    k_range = np.arange(1, 100)
    show_validation_curve(k_range, KNeighborsClassifier(), X_train, y_train, scoring="accuracy", hyper_param="n_neighbors", filename="plots\{}\KNN_valid_curve_k.png".format(dataset), dataset=dataset)

    param_grid = {'n_neighbors': np.arange(10, 101,10)}
    gs = GridSearchCV(KNeighborsClassifier(), param_grid, verbose=1, n_jobs=8, cv=5)
    grid_result = gs.fit(X_train, y_train)

    print("KNN best params found: " + str(grid_result.best_params_))

    clf_best = KNeighborsClassifier(n_neighbors=grid_result.best_params_['n_neighbors'])

    tstart = time.time()
    clf_best.fit(X_train, y_train)
    tend = time.time()
    train_time = tend - tstart
    tstart = time.time()
    y_pred = clf_best.predict(X_test)
    tend = time.time()
    test_time = tend - tstart
    dt_accuracy = accuracy_score(y_test, y_pred)

    y_pred_proba = clf_best.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    show_ROC(fpr, tpr, roc_auc, "\{}\KNN_ROC.png".format(dataset), dataset)

    if dataset=="heart":
        train_sizes = np.linspace(0.2, 1.0, 5)
    else:
        train_sizes = None

    show_learning_curve(clf_best, X_train, y_train, "plots\{}\KNN_learning_curve.png".format(dataset), dataset, train_sizes=train_sizes)

    print('DT: Fitting time (train data): %f seconds' % train_time)
    print('DT: Inference time (test data): %f seconds' % test_time)

    show_confusion_marix(clf_best, X_test, y_test, "plots\{}\KNN_confusion_matrix.png".format(dataset), dataset)

    return train_time, test_time, dt_accuracy


