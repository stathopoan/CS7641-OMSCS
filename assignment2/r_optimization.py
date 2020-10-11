import mlrose_hiive as mlrose
from mlrose_hiive import ExpDecay, simulated_annealing, random_hill_climb, genetic_alg, mimic
import numpy as np
import time

import model_problems
from utils import plot_objective_curve, plot_fitness_time_curve, plot_time_curve, plot_fitness_evaluation_curve


def optimize_with_mimic(problem, problem_dsc, max_iters, mimic_pop, mimic_keep_pct):
    print("Optimizing with MIMIC...")
    random_states = [3, 98, 654]

    fitting_times = np.zeros((len(random_states), max_iters + 1))
    curves = np.zeros((len(random_states), max_iters))
    evaluations = np.zeros((len(random_states), max_iters + 1))
    best_state = None
    best_objective = None

    for j, state in enumerate(random_states):
        iteration = [0]
        tstart = time.time()
        best_state, best_objective, fitness_curve = mimic(problem,
                                                          max_attempts=10,
                                                          max_iters=max_iters,
                                                          pop_size=mimic_pop,
                                                          keep_pct=mimic_keep_pct,
                                                          curve=True,
                                                          state_fitness_callback=my_callback,
                                                          callback_user_info=[fitting_times, j, tstart,
                                                                              evaluations, iteration],
                                                          random_state=state)

        curves[j, :len(fitness_curve)] = fitness_curve

    fitting_times = fitting_times[:, :-1]
    evaluations = evaluations[:, :-1]

    print("Best state found: {}. Fitness score: {}".format(best_state, best_objective))

    plot_objective_curve(curves, "mimic", filename="plots\\{}\\mimic_fitness_curve.png".format(problem_dsc))
    plot_fitness_time_curve(curves, fitting_times, "mimic",
                            filename="plots\\{}\\mimic_fitness_time_curve.png".format(problem_dsc))
    plot_time_curve(fitting_times, np.arange(max_iters), "mimic",
                    filename="plots\\{}\\mimic_time_curve.png".format(problem_dsc))
    plot_fitness_evaluation_curve(curves, evaluations, "mimic",
                                  filename="plots\\{}\\mimic_fitness_evaluation_curve.png".format(problem_dsc))

    curves_mean = np.nanmean(np.where(curves != 0, curves, np.nan), axis=0)
    curves_mean = ffnan(curves_mean)
    evaluations_mean = np.nanmean(np.where(evaluations != 0, evaluations, np.nan), axis=0)
    evaluations_mean = ffnan(evaluations_mean)
    times_mean = np.nanmean(np.where(fitting_times != 0, fitting_times, np.nan), axis=0)
    return curves_mean, evaluations_mean, times_mean


def optimize_with_ga(problem, problem_dsc, max_iters, ga_pop_size, ga_mutation_prob):
    print("Optimizing with GA...")
    random_states = [3, 98, 654]

    fitting_times = np.zeros((len(random_states), max_iters + 1))
    curves = np.zeros((len(random_states), max_iters))
    evaluations = np.zeros((len(random_states), max_iters + 1))
    best_state = None
    best_objective = None

    for j, state in enumerate(random_states):
        iteration = [0]
        tstart = time.time()
        best_state, best_objective, fitness_curve = genetic_alg(problem,
                                                                max_attempts=10,
                                                                max_iters=max_iters,
                                                                curve=True,
                                                                pop_size=ga_pop_size,
                                                                mutation_prob=ga_mutation_prob,
                                                                state_fitness_callback=my_callback,
                                                                callback_user_info=[fitting_times, j, tstart,
                                                                                    evaluations, iteration],
                                                                random_state=state)

        curves[j, :len(fitness_curve)] = fitness_curve

    fitting_times = fitting_times[:, :-1]
    evaluations = evaluations[:, :-1]

    print("Best state found: {}. Fitness score: {}".format(best_state, best_objective))

    plot_objective_curve(curves, "ga", filename="plots\\{}\\ga_fitness_curve.png".format(problem_dsc))
    plot_fitness_time_curve(curves, fitting_times, "ga",
                            filename="plots\\{}\\sa_fitness_time_curve.png".format(problem_dsc))
    plot_time_curve(fitting_times, np.arange(max_iters), "ga",
                    filename="plots\\{}\\sa_time_curve.png".format(problem_dsc))
    plot_fitness_evaluation_curve(curves, evaluations, "ga",
                                  filename="plots\\{}\\ga_fitness_evaluation_curve.png".format(problem_dsc))

    curves_mean = np.nanmean(np.where(curves != 0, curves, np.nan), axis=0)
    curves_mean = ffnan(curves_mean)
    evaluations_mean = np.nanmean(np.where(evaluations != 0, evaluations, np.nan), axis=0)
    evaluations_mean = ffnan(evaluations_mean)
    times_mean = np.nanmean(np.where(fitting_times != 0, fitting_times, np.nan), axis=0)
    return curves_mean, evaluations_mean, times_mean


def optimize_with_sa(problem, problem_dsc, max_iters, sa_init_temp, sa_exp_const, sa_min_temp):
    print("Optimizing with SA...")
    random_states = [3, 98, 654]

    exp_decay = ExpDecay(init_temp=sa_init_temp,
                         exp_const=sa_exp_const,
                         min_temp=sa_min_temp)

    # fitting_times = np.zeros((len(random_states), len(iterations)))
    fitting_times = np.zeros((len(random_states), max_iters + 1))
    curves = np.zeros((len(random_states), max_iters))
    evaluations = np.zeros((len(random_states), max_iters + 1))
    best_state = None
    best_objective = None

    for j, state in enumerate(random_states):
        # print(state)
        iteration = [0]
        tstart = time.time()
        best_state, best_objective, fitness_curve = simulated_annealing(problem,
                                                                        schedule=exp_decay,
                                                                        max_attempts=10,
                                                                        max_iters=max_iters,
                                                                        curve=True,
                                                                        state_fitness_callback=my_callback,
                                                                        callback_user_info=[fitting_times, j, tstart,
                                                                                            evaluations, iteration],
                                                                        random_state=state)

        curves[j, :len(fitness_curve)] = fitness_curve

    fitting_times = fitting_times[:, :-1]
    evaluations = evaluations[:, :-1]

    print("Best state found: {}. Fitness score: {}".format(best_state, best_objective))

    plot_objective_curve(curves, "sa", filename="plots\\{}\\sa_fitness_curve.png".format(problem_dsc))
    plot_fitness_time_curve(curves, fitting_times, "sa",
                            filename="plots\\{}\\sa_fitness_time_curve.png".format(problem_dsc))
    plot_time_curve(fitting_times, np.arange(max_iters), "sa",
                    filename="plots\\{}\\sa_time_curve.png".format(problem_dsc))
    plot_fitness_evaluation_curve(curves, evaluations, "sa",
                                  filename="plots\\{}\\sa_fitness_evaluation_curve.png".format(problem_dsc))

    curves_mean = np.nanmean(np.where(curves != 0, curves, np.nan), axis=0)
    curves_mean = ffnan(curves_mean)
    evaluations_mean = np.nanmean(np.where(evaluations != 0, evaluations, np.nan), axis=0)
    evaluations_mean = ffnan(evaluations_mean)
    times_mean = np.nanmean(np.where(fitting_times != 0, fitting_times, np.nan), axis=0)
    return curves_mean, evaluations_mean, times_mean


def optimize_with_rhc(problem, problem_dsc, max_iters, rhc_num_restarts):
    print("Optimizing with RHC...")
    random_states = [3, 98, 654]

    fitting_times = np.zeros((len(random_states), max_iters + 1))
    curves = np.zeros((len(random_states), max_iters))
    evaluations = np.zeros((len(random_states), max_iters + 1))
    best_state = None
    best_objective = None

    for j, state in enumerate(random_states):
        iteration = [0]
        tstart = time.time()
        restart_no = [0]
        best_state, best_objective, fitness_curve = random_hill_climb(problem,
                                                                      max_attempts=1000,
                                                                      max_iters=max_iters,
                                                                      curve=True,
                                                                      restarts=rhc_num_restarts,
                                                                      state_fitness_callback=my_callback_rhc,
                                                                      callback_user_info=[fitting_times, j, tstart,
                                                                                          evaluations, iteration,
                                                                                          restart_no, rhc_num_restarts, max_iters],
                                                                      random_state=state)
        curves[j, :len(fitness_curve)] = fitness_curve
        global FOUR_PEAKS_NUM_EVAL_GLOB
        model_problems.FOUR_PEAKS_NUM_EVAL_GLOB = 0

    print("Best state found: {}. Fitness score: {}".format(best_state, best_objective))

    fitting_times = fitting_times[:, :-1]
    evaluations = evaluations[:, :-1]

    plot_objective_curve(curves, "rhc", filename="plots\\{}\\rhc_fitness_curve.png".format(problem_dsc))
    plot_fitness_time_curve(curves, fitting_times, "rhc",
                            filename="plots\\{}\\rhc_fitness_time_curve.png".format(problem_dsc))
    plot_time_curve(fitting_times, np.arange(max_iters), "rhc",
                    filename="plots\\{}\\rhc_time_curve.png".format(problem_dsc))
    plot_fitness_evaluation_curve(curves, evaluations, "rhc",
                                  filename="plots\\{}\\rhc_fitness_evaluation_curve.png".format(problem_dsc))

    curves_mean = np.nanmean(np.where(curves != 0, curves, np.nan), axis=0)
    # curves_mean = ffnan(curves_mean)
    evaluations_mean = np.nanmean(np.where(evaluations != 0, evaluations, np.nan), axis=0)
    # evaluations_mean = ffnan(evaluations_mean)
    times_mean = np.nanmean(np.where(fitting_times != 0, fitting_times, np.nan), axis=0)
    return curves_mean, evaluations_mean, times_mean


def my_callback_rhc(**kwargs):
    ft = kwargs["user_data"][0]
    state = kwargs["user_data"][1]
    tstart = kwargs["user_data"][2]
    evaluations = kwargs["user_data"][3]
    current_iteration = kwargs["user_data"][4]
    restart = kwargs["user_data"][5]
    num_restart = kwargs["user_data"][6]
    max_iters = kwargs["user_data"][7]

    if current_iteration[0] >= max_iters:
        return False

    evaluations[state, current_iteration[0]] = model_problems.FOUR_PEAKS_NUM_EVAL_GLOB
    ft[state, current_iteration[0]] = time.time() - tstart

    if 'done' in kwargs and kwargs.get("done") == True:
        if restart[0] < num_restart:
            restart[0] += 1
        # else:
            # global FOUR_PEAKS_NUM_EVAL_GLOB
            # model_problems.FOUR_PEAKS_NUM_EVAL_GLOB = 0

    current_iteration[0] += 1

    return True



def my_callback(**kwargs):
    current_iteration = kwargs["user_data"][4]
    ft = kwargs["user_data"][0]
    state = kwargs["user_data"][1]
    tstart = kwargs["user_data"][2]
    evaluations = kwargs["user_data"][3]
    evaluations[state, current_iteration[0]] = model_problems.FOUR_PEAKS_NUM_EVAL_GLOB
    ft[state, current_iteration[0]] = time.time() - tstart
    if 'done' in kwargs and kwargs.get("done") == True:
        global FOUR_PEAKS_NUM_EVAL_GLOB
        model_problems.FOUR_PEAKS_NUM_EVAL_GLOB = 0

    current_iteration[0] += 1
    return True


def ffnan(arr, axis=0):
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [np.arange(k)[tuple([slice(None) if dim == i else np.newaxis
                               for dim in range(len(arr.shape))])]
           for i, k in enumerate(arr.shape)]
    slc[axis] = idx
    return arr[tuple(slc)]
