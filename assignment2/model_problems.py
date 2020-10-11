from random import random

from mlrose_hiive import FourPeaks, DiscreteOpt, FlipFlop, Queens, MaxKColor, TravellingSales, TSPOpt, ContinuousPeaks
import numpy as np

from r_optimization import optimize_with_sa, optimize_with_rhc, optimize_with_ga, optimize_with_mimic
from utils import plot_aggregated_objective_curve, plot_aggregated_fitness_evaluation_curve, \
    plot_aggregated_fitness_time_curve, plot_aggregated_time_curve

FOUR_PEAKS_NUM_EVAL_GLOB = 0


def run_problem(problem_dsc):
    global FOUR_PEAKS_NUM_EVAL_GLOB
    FOUR_PEAKS_NUM_EVAL_GLOB = 0
    print(".... {} ....".format(problem_dsc))
    problem = None
    max_iters = 3000
    curves = np.zeros((4, max_iters))
    evaluations = np.zeros((4, max_iters))
    times = np.zeros((4, max_iters))
    algorithms = ["rhc", "sa", "ga", "mimic"]
    problem = None

    # Default Hyper params
    sa_init_temp = 100
    sa_exp_const = 0.1
    sa_min_temp = 0.001
    rhc_num_restarts = 10
    mimic_pop = 100
    mimic_keep_pct = 0.1
    ga_pop_size = 100
    ga_mutation_prob = 0.1

    if problem_dsc == "4_peaks":
        # Run optimizations
        mimic_pop = 300
        mimic_keep_pct = 0.2
        ga_pop_size = 200
        ga_mutation_prob = 0.20
        sa_init_temp = 100
        sa_exp_const = 0.1
        sa_min_temp = 0.001
        fitness_cust = MyFourPeaks(t_pct=0.10)
        problem = DiscreteOpt(length=100, fitness_fn=fitness_cust, maximize=True, max_val=2)

    elif problem_dsc == "flip_flop":
        # Run optimizations
        ga_pop_size = 150
        ga_mutation_prob = 0.1
        mimic_pop = 500
        mimic_keep_pct = 0.2
        sa_init_temp = 50
        sa_exp_const = 0.1
        sa_min_temp = 0.001
        fitness_cust = MyFlipFlop()
        problem = DiscreteOpt(length=100, fitness_fn=fitness_cust, maximize=True, max_val=2)

    elif problem_dsc == "continous_peaks":
        # Run optimizations
        ga_pop_size = 300
        ga_mutation_prob = 0.1
        mimic_pop = 500
        mimic_keep_pct = 0.2
        sa_init_temp = 100
        sa_exp_const = 0.1
        sa_min_temp = 1
        fitness_cust = MyContinuousPeaks(t_pct=0.10)
        problem = DiscreteOpt(length=100, fitness_fn=fitness_cust, maximize=True, max_val=2)

    elif problem_dsc == "kcolors":
        # Run optimizations
        np.random.seed(30)
        edges = create_max_k_color_edges(max_nodes=50, max_edges=1000)
        fitness_cust = MyMaxKColor(edges=edges)
        problem = DiscreteOpt(length=50, fitness_fn=fitness_cust, maximize=False, max_val=10)
    elif problem_dsc == "tsp":
        # Run optimizations
        cities_distances = create_distancies(distance_matrix)
        fitness_cust = MyTravellingSales(distances=cities_distances)
        problem = TSPOpt(length=12, fitness_fn=fitness_cust, maximize=True)

    curves[0, :], evaluations[0, :], times[0, :] = optimize_with_rhc(problem, problem_dsc, max_iters, rhc_num_restarts=rhc_num_restarts)
    curves[1, :], evaluations[1, :], times[1, :] = optimize_with_sa(problem, problem_dsc, max_iters, sa_init_temp=sa_init_temp, sa_exp_const=sa_exp_const, sa_min_temp=sa_min_temp)
    curves[2, :], evaluations[2, :], times[2, :] = optimize_with_ga(problem, problem_dsc, max_iters, ga_pop_size = ga_pop_size, ga_mutation_prob = ga_mutation_prob)
    curves[3, :], evaluations[3, :], times[3, :] = optimize_with_mimic(problem, problem_dsc, max_iters, mimic_pop=mimic_pop, mimic_keep_pct=mimic_keep_pct)

    plot_aggregated_objective_curve(curves, algorithms, filename="plots\\{}\\agg_fitness_curve.png".format(problem_dsc),
                                    problem_dsc=problem_dsc)
    plot_aggregated_fitness_evaluation_curve(curves, evaluations, algorithms,
                                             filename="plots\\{}\\agg_fitness_evaluation_curve.png".format(problem_dsc),
                                             problem_dsc=problem_dsc)
    plot_aggregated_fitness_time_curve(curves, times, algorithms,
                                       filename="plots\\{}\\agg_fitness_time_curve.png".format(problem_dsc),
                                       problem_dsc=problem_dsc)
    plot_aggregated_time_curve(times, algorithms, filename="plots\\{}\\agg_time_curve.png".format(problem_dsc),
                               problem_dsc=problem_dsc)
    return problem


class MyFourPeaks(FourPeaks):
    def __init__(self, t_pct=0.1):
        global FOUR_PEAKS_NUM_EVAL_GLOB
        FOUR_PEAKS_NUM_EVAL_GLOB = 0
        super().__init__(t_pct)

    def evaluate(self, state):
        global FOUR_PEAKS_NUM_EVAL_GLOB
        FOUR_PEAKS_NUM_EVAL_GLOB = FOUR_PEAKS_NUM_EVAL_GLOB + 1
        return super().evaluate(state)


class MyFlipFlop(FlipFlop):
    def __init__(self):
        global FOUR_PEAKS_NUM_EVAL_GLOB
        FOUR_PEAKS_NUM_EVAL_GLOB = 0
        super().__init__()

    def evaluate(self, state):
        global FOUR_PEAKS_NUM_EVAL_GLOB
        FOUR_PEAKS_NUM_EVAL_GLOB = FOUR_PEAKS_NUM_EVAL_GLOB + 1
        return super().evaluate(state)


class MyQueens(Queens):
    def __init__(self):
        global FOUR_PEAKS_NUM_EVAL_GLOB
        FOUR_PEAKS_NUM_EVAL_GLOB = 0
        super().__init__()

    def evaluate(self, state):
        global FOUR_PEAKS_NUM_EVAL_GLOB
        FOUR_PEAKS_NUM_EVAL_GLOB = FOUR_PEAKS_NUM_EVAL_GLOB + 1
        return super().evaluate(state)


class MyMaxKColor(MaxKColor):
    def __init__(self, edges):
        global FOUR_PEAKS_NUM_EVAL_GLOB
        FOUR_PEAKS_NUM_EVAL_GLOB = 0
        super().__init__(edges=edges)

    def evaluate(self, state):
        global FOUR_PEAKS_NUM_EVAL_GLOB
        FOUR_PEAKS_NUM_EVAL_GLOB = FOUR_PEAKS_NUM_EVAL_GLOB + 1
        return super().evaluate(state)

class MyContinuousPeaks(ContinuousPeaks):
    def __init__(self, t_pct=0.1):
        global FOUR_PEAKS_NUM_EVAL_GLOB
        FOUR_PEAKS_NUM_EVAL_GLOB = 0
        super().__init__(t_pct=t_pct)

    def evaluate(self, state):
        global FOUR_PEAKS_NUM_EVAL_GLOB
        FOUR_PEAKS_NUM_EVAL_GLOB = FOUR_PEAKS_NUM_EVAL_GLOB + 1
        return super().evaluate(state)


def create_max_k_color_edges(max_nodes, max_edges):
    edges = []
    while len(edges) < max_edges:
        new_edge = create_Edge(max_nodes)
        if new_edge not in edges:
            edges.append(new_edge)
    return edges


def create_Edge(max_nodes):
    return (np.random.randint(1, max_nodes), np.random.randint(1, max_nodes))


class MyTravellingSales(TravellingSales):
    def __init__(self, coords=None, distances=None):
        global FOUR_PEAKS_NUM_EVAL_GLOB
        FOUR_PEAKS_NUM_EVAL_GLOB = 0
        super().__init__(coords=coords, distances=distances)

    def evaluate(self, state):
        global FOUR_PEAKS_NUM_EVAL_GLOB
        FOUR_PEAKS_NUM_EVAL_GLOB = FOUR_PEAKS_NUM_EVAL_GLOB + 1
        return super().evaluate(state)


# Reference: https://developers.google.com/optimization/routing/tsp
# 0. New York - 1. Los Angeles - 2. Chicago - 3. Minneapolis - 4. Denver - 5. Dallas - 6. Seattle - 7. Boston - 8. San Francisco - 9. St. Louis - 10. Houston - 11. Phoenix - 12. Salt Lake City
distance_matrix = [
    [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
    [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
    [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
    [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
    [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
    [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
    [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
    [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
    [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
    [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
    [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
    [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
    [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
]


def create_distancies(matrix):
    cities_distances = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if i == j:
                continue
            new_distance = (i, j, matrix[i][j])
            cities_distances.append(new_distance)
    return cities_distances
