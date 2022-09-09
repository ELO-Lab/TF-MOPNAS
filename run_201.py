import argparse
import logging
import os
import pickle as p
import sys
import time
from copy import deepcopy
from nats_bench import create
import matplotlib.pyplot as plt
import numpy as np

from utils import get_model_infos
from pruning_models.model_201 import *
from utils import calculate_IGD_value, set_seed
from utils import get_front_0
from training_free_metrics import get_config_for_training_free_calculator, get_training_free_calculator

xshape = (1, 3, 32, 32)
nClasses = 10
dataset = 'CIFAR-10'
OPS_LIST = {
    '0': 'none',
    '1': 'skip_connect',
    '2': 'nor_conv_1x1',
    '3': 'nor_conv_3x3',
    '4': 'avg_pool_3x3'
}


def convert_keep_mask_2_arch_str(keep_mask):
    arch_str = ''
    for i in range(len(keep_mask)):
        for j in range(len(keep_mask[i])):
            if keep_mask[i][j]:
                arch_str += f'{j}'
    return arch_str


def convert_to_ori_api_input(arch_str):
    ori_input = f'|{OPS_LIST[arch_str[0]]}~0|+' \
                f'|{OPS_LIST[arch_str[1]]}~0|{OPS_LIST[arch_str[2]]}~1|+' \
                f'|{OPS_LIST[arch_str[3]]}~0|{OPS_LIST[arch_str[4]]}~1|{OPS_LIST[arch_str[5]]}~2|'
    return ori_input


def evaluate(final_opt_archs, api, pf, path_results):
    approximation_set = []
    approximation_front = []
    total_pos_training_time = 0.0
    best_arch, best_test_acc = None, 0.0
    for keep_mask in final_opt_archs:
        arch = convert_to_ori_api_input(convert_keep_mask_2_arch_str(keep_mask))
        idx = api.query_index_by_arch(arch)  # query the index of architecture in database

        info = api.get_more_info(idx, dataset='cifar10', hp='200', is_random=False)
        cost = api.get_cost_info(idx, dataset='cifar10', hp='200')

        FLOPs = cost['flops'] / 1e3
        test_error = 1.0 - info['test-accuracy'] / 100
        F = [FLOPs, test_error]
        total_pos_training_time += info['train-per-time'] * 200

        if info['test-accuracy'] > best_test_acc:
            best_arch = arch
            best_test_acc = info['test-accuracy']

        approximation_set.append(arch)
        approximation_front.append(F)
    approximation_front = np.round(np.array(approximation_front), 4)
    idx_front_0 = get_front_0(approximation_front)

    approximation_set = np.array(approximation_set)[idx_front_0]
    approximation_front = approximation_front[idx_front_0]
    IGD = np.round(calculate_IGD_value(pareto_front=pf, non_dominated_front=approximation_front), 6)

    logging.info(f'Evaluate -> Done!\n')
    logging.info(f'IGD: {IGD}')
    logging.info(f'Best Architecture: {best_arch}')
    logging.info(f'Best Architecture (performance): {np.round(best_test_acc, 2)}\n')
    rs = {
        'n_archs_evaluated': len(final_opt_archs),
        'approximation_set': approximation_set,
        'approximation_front': approximation_front,
        'total_pos_training_time': total_pos_training_time,
        'best_arch_found': best_arch,
        'best_arch_found (performance)': np.round(best_test_acc, 2),
        'IGD': IGD,
    }
    p.dump(rs, open(f'{path_results}/results_evaluation.p', 'wb'))

    logging.info(f'--- Approximation set ---\n')
    for i in range(len(approximation_set)):
        logging.info(
            f'arch: {approximation_set[i]} - FLOPs: {approximation_front[i][0]} - testing error: {approximation_front[i][1]}\n')
    plt.scatter(approximation_front[:, 0], approximation_front[:, 1], facecolors='blue', s=30,
                label='Approximation front')
    plt.scatter(pf[:, 0], pf[:, 1], edgecolors='red', facecolors='none', s=60, label='Pareto-optimal front')
    plt.legend()
    plt.title(f'NAS-Bench-201, {dataset}, Synflow')
    plt.savefig(f'{path_results}/approximation_front.jpg')
    plt.clf()
    return IGD, np.round(best_test_acc, 2)


def prune(tf_ind, path_data, path_results, seed):
    list_arch_parent = [
        [[True, True, True, True, True],
         [True, True, True, True, True],
         [True, True, True, True, True],
         [True, True, True, True, True],
         [True, True, True, True, True],
         [True, True, True, True, True]]
    ]  # At beginning, activating all operations
    max_nPrunes = len(list_arch_parent[-1])
    i = 0

    config = get_config_for_training_free_calculator(search_space='NASBench201', dataset=dataset,
                                                     seed=seed, path_data=path_data)
    tf_calculator = get_training_free_calculator(config=config, method_type=tf_ind)

    id_arch = 0
    while i <= max_nPrunes - 1:
        logging.info(f'------- The {i + 1}-th pruning -------\n')
        list_arch_child = []
        F_arch_child = []
        for arch in list_arch_parent:
            arch_child = deepcopy(arch)
            arch_child[i] = [False, False, False, False, False]
            for j in range(len(arch_child[i])):
                id_arch += 1
                arch_child[i][j] = True
                list_arch_child.append(deepcopy(arch_child))

                flat_list = [item for sublist in list_arch_child[-1] for item in sublist]
                network = get_model_from_arch_str(flat_list, nClasses)

                flop, param = get_model_infos(network, xshape)
                tf_metric_value = tf_calculator.compute(keep_mask=flat_list)[tf_ind]
                F = [flop, -tf_metric_value]
                logging.info(f'ID Arch: {id_arch}')
                logging.info(f'Keep mask:\n{list_arch_child[-1]}')
                logging.info(f'FLOPs: {F[0]}')
                logging.info(f'Synflow: {F[-1]}\n')
                F_arch_child.append(F)
                arch_child[i][j] = False
        idx_front_0 = get_front_0(F_arch_child)
        list_arch_parent = np.array(deepcopy(list_arch_child))[idx_front_0]
        logging.info(f'Number of architectures for the next pruning: {len(list_arch_parent)}\n')
        i += 1

    final_opt_archs = list_arch_parent
    p.dump(final_opt_archs, open(f'{path_results}/pruning_results.p', 'wb'))
    return final_opt_archs


def main(kwargs):
    n_runs = kwargs.n_runs
    init_seed = kwargs.seed
    random_seeds_list = [init_seed + run * 100 for run in range(n_runs)]

    if kwargs.path_data is None:
        path_data = './benchmark_data'
    else:
        path_data = kwargs.path_data
    if kwargs.path_results is None:
        path_results = './results/201/TF-MOPNAS'
    else:
        path_results = kwargs.path_results
    tf_metric = 'synflow'  # can be another training-free metric, i.e., jacov, snip, grad_norm, etc.

    api = create(f'{path_data}/NASBench201/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)
    pareto_opt_front = p.load(open(f'{path_data}/NASBench201/[{dataset}]_pareto_front(testing).p', 'rb'))
    pareto_opt_front[:, 0] /= 1e3

    logging.info(f'******* PROBLEM *******')
    logging.info(f'- Benchmark: NAS-Bench-201')
    logging.info(f'- Dataset: CIFAR-10\n')

    logging.info(f'******* RUNNING *******')
    logging.info(f'- Pruning:')
    logging.info(f'\t+ The first objective (minimize): FLOPs')
    logging.info(f'\t+ The second objective (minimize): -Synflow')

    logging.info(f'- Evaluate:')
    logging.info(f'\t+ The first objective (minimize): FLOPs')
    logging.info(f'\t+ The second objective (minimize): test error\n')

    logging.info(f'******* ENVIRONMENT *******')
    logging.info(f'- Path for saving results: {path_results}\n')

    final_IGD_lst = []
    best_acc_found_lst = []

    for run_i in range(n_runs):
        logging.info(f'Run ID: {run_i + 1}')
        sub_path_results = path_results + '/' + f'{run_i}'

        try:
            os.mkdir(sub_path_results)
        except FileExistsError:
            pass
        logging.info(f'Path for saving results: {sub_path_results}')

        random_seed = random_seeds_list[run_i]
        logging.info(f'Random seed: {run_i}')
        set_seed(random_seed)

        s = time.time()
        final_opt_archs = prune(tf_ind=tf_metric, path_data=path_data, path_results=sub_path_results, seed=random_seed)
        executed_time = time.time() - s
        logging.info(f'Prune Done! Execute in {executed_time} seconds.\n')
        p.dump(executed_time, open(f'{sub_path_results}/running_time.p', 'wb'))

        IGD, best_acc = evaluate(final_opt_archs=final_opt_archs, api=api, pf=pareto_opt_front,
                                 path_results=sub_path_results)
        final_IGD_lst.append(IGD)
        best_acc_found_lst.append(best_acc)

    logging.info(f'Average IGD: {np.round(np.mean(final_IGD_lst), 4)} ({np.round(np.std(final_IGD_lst), 4)})')
    logging.info(
        f'Average best test-accuracy: {np.round(np.mean(best_acc_found_lst), 4)} ({np.round(np.std(best_acc_found_lst), 4)})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' ENVIRONMENT '''
    parser.add_argument('--path_data', type=str, default=None, help='path for loading data')
    parser.add_argument('--path_results', type=str, default=None, help='path for saving results')
    parser.add_argument('--n_runs', type=int, default=31, help='number of experiment runs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    main(args)
