from mdp_toolbox_test_policy import run_vi_pi_forest_experiment, run_q_learning_forest_experiment
from q_learning_experiment import run_Q_learning_experiment
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
import numpy as np
from hiive.mdptoolbox.example import forest

from vi_pi_experiment import run_vi_pi_experiment


def run_experiments():
    np.random.seed(32)
    MAPS = {
        "4x4": generate_random_map(4),
        "8x8": generate_random_map(8)
    }
    frozen_lake_env_4 = FrozenLakeEnv(desc=MAPS["4x4"])
    frozen_lake_env_8 = FrozenLakeEnv(desc=MAPS["8x8"])
    envs = [frozen_lake_env_4, frozen_lake_env_8]
    run_Q_learning_experiment(envs, MAPS)
    run_vi_pi_experiment(envs, MAPS)
    P, R = forest(S=20, r1=10, r2=6, p=0.1)
    run_vi_pi_forest_experiment(P, R, 20)
    run_q_learning_forest_experiment(P,R,20)
    P, R = forest(S=500, r1=10, r2=6, p=0.1)
    run_vi_pi_forest_experiment(P, R, 500)
    run_q_learning_forest_experiment(P, R, 500)

if __name__ == '__main__':
    run_experiments()
