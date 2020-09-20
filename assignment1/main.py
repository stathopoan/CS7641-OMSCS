from Boost_experiment import run_AdaBoost_learner
from DT_experiment import run_DT_learner
from KNN_experiment import run_KNN
from NN_experiment import run_NN
from SVM_experiment import run_SVM_learner
from explore_data import explore_data_wine, explore_data_heart_disease
import numpy as np

from utils import show_metrics


def start_procedure():
    fitting_time_wine = np.zeros(5)
    inference_time_wine = np.zeros(5)
    accuracy_wine = np.zeros(5)

    fitting_time_heart = np.zeros(5)
    inference_time_heart = np.zeros(5)
    accuracy_heart = np.zeros(5)

    X_wine,y_wine = explore_data_wine()
    X_heart,y_heart = explore_data_heart_disease()
    fitting_time_wine[0], inference_time_wine[0], accuracy_wine[0] = run_DT_learner(X_wine,y_wine, "wine")
    fitting_time_wine[1], inference_time_wine[1], accuracy_wine[1] = run_AdaBoost_learner(X_wine,y_wine, "wine")
    fitting_time_wine[2], inference_time_wine[2], accuracy_wine[2]= run_SVM_learner(X_wine,y_wine, "wine")
    fitting_time_wine[3], inference_time_wine[3], accuracy_wine[3] = run_KNN(X_wine,y_wine, "wine")
    fitting_time_wine[4], inference_time_wine[4], accuracy_wine[4] = run_NN(X_wine, y_wine, "wine")

    fitting_time_heart[0], inference_time_heart[0], accuracy_heart[0] = run_DT_learner(X_heart,y_heart, "heart")
    fitting_time_heart[1], inference_time_heart[1], accuracy_heart[1] = run_AdaBoost_learner(X_heart, y_heart, "heart")
    fitting_time_heart[2], inference_time_heart[2], accuracy_heart[2] = run_SVM_learner(X_heart,y_heart, "heart")
    fitting_time_heart[3], inference_time_heart[3], accuracy_heart[3] = run_KNN(X_heart,y_heart, "heart")
    fitting_time_heart[4], inference_time_heart[4], accuracy_heart[4] = run_NN(X_heart, y_heart, "heart")

    show_metrics(fitting_time_wine, inference_time_wine, accuracy_wine, fitting_time_heart, inference_time_heart, accuracy_heart)

if __name__ == '__main__':
    start_procedure()