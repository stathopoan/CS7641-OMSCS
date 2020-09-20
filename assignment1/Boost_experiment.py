import time

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, KFold, \
    GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

from utils import show_learning_curve, show_ROC, show_confusion_marix


def run_AdaBoost_learner(X, y, dataset):
    print("...Boosting LEARNER....")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9098, shuffle=True)

    dt = DecisionTreeClassifier(ccp_alpha=0.001, random_state=0)

    clf = AdaBoostClassifier(base_estimator=dt, n_estimators=10, random_state=0)
    clf.fit(X_train, y_train)

    # evaluate the model
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    n_scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

    # Use a very basic tree and create and Adaboost classifier to test the accuracy in repset of the number of learners
    n_estimators = 2000
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1, random_state=0),
                             n_estimators=n_estimators)

    validation_curve(clf, X_train, y_train, "plots\{}\Adaboost_valid_curve_estimators.png".format(dataset), dataset)

    n_estimators = 1000
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(ccp_alpha=0.001, random_state=0),
                             n_estimators=n_estimators)

    validation_curve(clf, X_train, y_train, "plots\{}\Adaboost_valid_curve_pruned_estimators.png".format(dataset), dataset)

    # Search for best hyper parameters regarding number of estimators and learning rate
    # define the grid of values to search
    grid = dict()
    grid['n_estimators'] = [200 ,300, 350, 400]
    grid['learning_rate'] = [0.001, 0.01, 0.1, 1.0]

    cv = KFold(n_splits=5, random_state=1)

    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1, random_state=0),
                             n_estimators=n_estimators)
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

    grid_result = grid_search.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


    clf_best = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, min_samples_leaf=1, random_state=0),
                             n_estimators=grid_result.best_params_['n_estimators'], learning_rate=grid_result.best_params_['learning_rate'],
                                  random_state=0)

    tstart = time.time()
    clf_best.fit(X_train,y_train)
    tend = time.time()
    train_time = tend - tstart
    show_learning_curve(clf_best, X_train, y_train, "plots\{}\Adaboost_learning curve.png".format(dataset), dataset)

    tstart = time.time()
    y_pred = clf_best.predict(X_test)
    tend = time.time()
    test_time = tend - tstart
    y_pred_proba = clf_best.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    show_ROC(fpr, tpr, roc_auc, "\{}\Adaboost_ROC_final.png".format(dataset), dataset)
    accuracy = accuracy_score(y_test, y_pred)

    show_confusion_marix(clf_best, X_test, y_test, "plots\{}\Adaboost_confusion_matrix.png".format(dataset), dataset)

    return train_time, test_time, accuracy


def validation_curve(clf, X_train, y_train, filename, dataset):
    n_splits = 5
    n_estimators = clf.get_params()["n_estimators"]
    train_scores = np.zeros((n_splits, n_estimators))
    test_scores = np.zeros((n_splits, n_estimators))
    cv = KFold(n_splits=n_splits, random_state=1, shuffle=True)
    i = 0
    for train_index, test_index in cv.split(X_train, y_train):
        clf.fit(X_train[train_index], y_train[train_index])
        train_score = list(clf.staged_score(X_train[train_index], y_train[train_index]))
        test_score = list(clf.staged_score(X_train[test_index], y_train[test_index]))
        train_scores[i, :] = train_score
        test_scores[i, :] = test_score

        i += 1


    train_scores_mean = np.mean(train_scores, axis=0)
    train_scores_std = np.std(train_scores, axis=0)
    test_scores_mean = np.mean(test_scores, axis=0)
    test_scores_std = np.std(test_scores, axis=0)

    learners = np.arange(0, train_scores_mean.shape[0], 1)

    plt.figure()
    plt.title("Validation Curve - Dataset: {}".format(dataset))
    plt.xlabel(r"Weak learners")
    plt.ylabel(r"Score - accuracy")
    lw = 2
    plt.semilogx(learners, train_scores_mean, label="Training score", color="darkorange", lw=lw)
    plt.fill_between(learners, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(learners, test_scores_mean, label="Cross-validation score", color="navy", lw=lw)
    plt.fill_between(learners, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    # plt.show()
    plt.savefig(filename)
    plt.clf()
