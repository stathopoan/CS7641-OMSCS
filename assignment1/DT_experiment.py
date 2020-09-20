import time

from utils import show_ROC, show_validation_curve, show_learning_curve, show_confusion_marix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.metrics import accuracy_score, roc_curve, auc, plot_roc_curve, plot_confusion_matrix
from sklearn import tree
import numpy as np



def run_DT_learner(X, y, dataset):
    print("...DECISION TREE LEARNER....")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9098, shuffle=True)

    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred)

    print("accuracy of default model: {}".format(dt_accuracy))

    y_pred_proba = clf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)


    show_ROC(fpr, tpr, roc_auc, "\{}\DT_ROC_default.png".format(dataset), dataset)

    param_range = np.arange(1, 21)

    show_validation_curve(param_range, clf, X_train, y_train, scoring="accuracy", hyper_param="max_depth", filename="plots\\{}\\DT_validation_curve.png".format(dataset), dataset=dataset)

    best_alpha = DT_pruning(X_train, y_train, X_test, y_test, clf, dataset)

    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha)
    tstart = time.time()
    clf.fit(X_train, y_train)
    tend = time.time()
    train_time = tend - tstart
    tstart = time.time()
    y_pred = clf.predict(X_test)
    tend = time.time()
    test_time = tend - tstart
    dt_accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy of pruned model: {}".format(dt_accuracy))

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    show_ROC(fpr, tpr, roc_auc, "\{}\DT_ROC_pruned.png".format(dataset), dataset)

    show_learning_curve(clf, X_train, y_train, "plots\{}\DT_learning_curve.png".format(dataset), dataset)

    print('DT: Fitting time (train data): %f seconds' % train_time)
    print('DT: Inference time (test data): %f seconds' % test_time)

    show_confusion_marix(clf, X_test, y_test, "plots\{}\DT_confusion_matrix.png".format(dataset), dataset)

    return train_time, test_time, dt_accuracy

# Reference: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
def DT_pruning(X_train, y_train, X_test, y_test, clf, dataset):
    path = tree.DecisionTreeClassifier(random_state=980234).cost_complexity_pruning_path(X=X_train, y=y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=980234, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)


    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    plt.savefig("plots\{}\DT_pruning_alpha.png".format(dataset))
    plt.clf()

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    best_alpha = ccp_alphas[np.argmax(test_scores)]
    # In the wine dataset from the diagrams the alpha value at the start is greater than the rest of ccp_alpha but
    # it is mostly due to noise. So after that the best value is selected
    if best_alpha==0:
        best_alpha = 0.001

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    ax.legend()
    plt.savefig("plots\{}\DT_pruning_select_alpha.png".format(dataset))
    plt.clf()

    return best_alpha