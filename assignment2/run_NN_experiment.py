from NN_GA import run_GA
from NN_RHC import run_RHC
from NN_SA import run_SA
from gradient_descent import run_NN
import numpy as np
from explore_data import explore_data_heart_disease
from utils import show_metrics


def start_procedure():
    fitting_time_heart = np.zeros(4)
    accuracy_heart = np.zeros(4)

    X_heart, y_heart = explore_data_heart_disease()

    accuracy_heart[0], fitting_time_heart[0] = run_NN(X_heart, y_heart)
    accuracy_heart[1], fitting_time_heart[1] = run_RHC(X_heart, y_heart)
    accuracy_heart[2], fitting_time_heart[2] = run_SA(X_heart, y_heart)
    accuracy_heart[3], fitting_time_heart[3] = run_GA(X_heart, y_heart)

    show_metrics(fitting_time_heart,accuracy_heart)


if __name__ == '__main__':
    start_procedure()