import time

from sklearn import svm, preprocessing
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils import show_validation_curve, show_learning_curve, show_ROC, show_confusion_marix


def run_SVM_learner(X, y, dataset):
    print("...SVM LEARNER....")

    # Scale the X data otherwise hyper params will go crazy
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    X = scaling.transform(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9098, shuffle=True)

    clf_poly = svm.SVC(kernel='poly', random_state=3)
    clf_poly.fit(X_train, y_train)

    C_range = [0.01, 0.1, 0.5, 1, 10, 50, 100, 500]

    show_validation_curve(C_range, clf_poly, X_train, y_train, scoring="accuracy", hyper_param="C", filename="plots\{}\SVM_poly_kernel_valid_curve_C.png".format(dataset), dataset=dataset)

    clf_rbf = svm.SVC(kernel='rbf', random_state=3)
    clf_rbf.fit(X_train, y_train)

    show_validation_curve(C_range, clf_rbf, X_train, y_train, scoring="accuracy", hyper_param="C", filename="plots\{}\SVM_rbf_kernel_valid_curve_C.png".format(dataset), dataset=dataset)

    param_grid = {'C': [0.01, 0.1, 7, 10, 15, 20]}
    clf_poly = GridSearchCV(clf_poly, param_grid, verbose=1, n_jobs=8, cv=5)
    clf_poly.fit(X_train, y_train)

    clf_rbf = GridSearchCV(clf_rbf, param_grid, verbose=1, n_jobs=8, cv=5)
    clf_rbf.fit(X_train, y_train)

    print("poly SVM best params found: " + str(clf_poly.best_params_))
    print("rbf SVM best params found: " + str(clf_rbf.best_params_))


    clf_poly = svm.SVC(kernel='poly', C=clf_poly.best_params_['C'], probability=True)
    tstart = time.time()
    clf_poly.fit(X_train, y_train)
    tend = time.time()
    train_time_poly = tend - tstart
    show_learning_curve(clf_poly, X_train, y_train, "plots\{}\SVM_poly_learning_curve.png".format(dataset), dataset)

    clf_rbf = svm.SVC(kernel='rbf', C=clf_rbf.best_params_['C'], probability=True)
    tstart = time.time()
    clf_rbf.fit(X_train, y_train)
    tend = time.time()
    train_time_rbf = tend - tstart
    show_learning_curve(clf_rbf, X_train, y_train, "plots\{}\SVM_rbf_learning_curve.png".format(dataset), dataset)

    # To show ROC you need to set probability=True in SVM instance which takes way too long time so disabled
    y_pred_proba = clf_poly.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    show_ROC(fpr, tpr, roc_auc, "\{}\SVM_ROC_poly.png".format(dataset), dataset)

    y_pred_proba = clf_rbf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    show_ROC(fpr, tpr, roc_auc, "\{}\SVM_ROC_rbf.png".format(dataset), dataset)

    show_confusion_marix(clf_poly, X_test, y_test, "plots\{}\SVM_poly_confusion_matrix.png".format(dataset), dataset)
    show_confusion_marix(clf_rbf, X_test, y_test, "plots\{}\SVM_rbf_confusion_matrix.png".format(dataset), dataset)

    tstart = time.time()
    y_pred = clf_poly.predict(X_test)
    tend = time.time()
    test_time_poly = tend-tstart
    accuracy_poly = accuracy_score(y_test, y_pred)
    tstart = time.time()
    y_pred = clf_rbf.predict(X_test)
    tend = time.time()
    test_time_rbf = tend - tstart
    accuracy_rbf = accuracy_score(y_test, y_pred)

    print('SVM poly: Fitting time (train data): %f seconds' % train_time_poly)
    print('SVM poly: Inference time (test data): %f seconds' % test_time_poly)
    print('SVM poly: Accuracy: %f' % accuracy_poly)

    print('SVM rbf: Fitting time (train data): %f seconds' % train_time_rbf)
    print('SVM rbf: Inference time (test data): %f seconds' % test_time_rbf)
    print('SVM rbf: Accuracy: %f' % accuracy_rbf)

    if accuracy_rbf > accuracy_poly:
        return train_time_rbf, test_time_rbf, accuracy_rbf
    else:
        return train_time_poly, test_time_poly, accuracy_poly
