import argparse
import logging
import os
import pickle as p
import sys
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from utils import get_model_infos
from pruning_models.model_201 import *
from utils import calculate_IGD_value, set_seed
from utils import get_front_0
from zero_cost_methods import get_config_for_zero_cost_predictor, get_zero_cost_predictor

xshape = (1, 3, 32, 32)
nClasses = 10
dataset = 'CIFAR-10'


def convert_keep_mask_2_arch_str(keep_mask):
    arch_str = ''
    for i in range(len(keep_mask)):
        for j in range(len(keep_mask[i])):
            if keep_mask[i][j]:
                arch_str += f'{j}'
    return arch_str


def mo_prune(tf_ind, path_data, path_results, seed):
    list_arch_parent = [[[True, True, True, True, True],
                         [True, True, True, True, True],
                         [True, True, True, True, True],
                         [True, True, True, True, True],
                         [True, True, True, True, True],
                         [True, True, True, True, True]]]
    max_nPrunes = len(list_arch_parent[-1])
    i = 0

    config = get_config_for_zero_cost_predictor(search_space='NASBench201', dataset=dataset,
                                                seed=seed, path_data=path_data)
    ZC_predictor = get_zero_cost_predictor(config=config, method_type=tf_ind)

    id_arch = 0
    while i <= max_nPrunes - 1:
        logging.info(f'Number of pruning: {i + 1}\n')
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
                tf_metric_value = ZC_predictor.query__(keep_mask=flat_list)[tf_ind]
                F = [flop, -tf_metric_value]
                logging.info(f'ID Arch: {id_arch}')
                logging.info(f'Keep mask:\n{list_arch_child[-1]}')
                logging.info(f'FLOPs: {F[0]}')
                logging.info(f'Synflow: {F[-1]}\n')
                F_arch_child.append(F)
                arch_child[i][j] = False
        idx_front_0 = get_front_0(F_arch_child)
        list_arch_parent = np.array(deepcopy(list_arch_child))[idx_front_0]
        logging.info(f'Number of pruning architectures: {len(list_arch_parent)}\n')
        i += 1

    p.dump(list_arch_parent, open(f'{path_results}/after_prune_result.p', 'wb'))
    return list_arch_parent


def evaluate(list_prune_arch, data, pf, path_results):
    final_list_arch = []
    final_list_F_arch = []
    total_train_time = 0.0
    for keep_mask in list_prune_arch:
        arch_str = convert_keep_mask_2_arch_str(keep_mask)
        FLOPs = data['200'][arch_str]['FLOPs'] / 1e3
        test_error = 1 - data['200'][arch_str]['test_acc'][-1]
        F = [FLOPs, test_error]
        total_train_time += data['200'][arch_str]['train_time'] * 200
        final_list_arch.append(arch_str)
        final_list_F_arch.append(F)
    final_list_F_arch = np.round(np.array(final_list_F_arch), 4)
    idx_front_0 = get_front_0(final_list_F_arch)

    final_list_arch = np.array(final_list_arch)[idx_front_0]
    final_list_F_arch = final_list_F_arch[idx_front_0]
    IGD = np.round(calculate_IGD_value(pareto_front=pf, non_dominated_front=final_list_F_arch), 6)
    rs = {
        'n_archs_evaluated': len(list_prune_arch),
        'arch_lst': final_list_arch,
        'F_lst': final_list_F_arch,
        'total_train_time': total_train_time,
        'IGD': IGD
    }
    p.dump(rs, open(f'{path_results}/results_(evaluation).p', 'wb'))

    plt.scatter(final_list_F_arch[:, 0], final_list_F_arch[:, 1], facecolors='blue', s=30, label='approximation front')
    plt.scatter(pf[:, 0], pf[:, 1], edgecolors='red', facecolors='none', s=60, label='pareto front')
    plt.legend()
    plt.title(f'NAS-Bench-201, {dataset}, Synflow')
    plt.savefig(f'{path_results}/approximation_front.jpg')
    plt.clf()
    logging.info(f'Evaluate Done! IGD: {IGD}')


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
    tf_metric = 'synflow'

    evaluate_mode = bool(kwargs.evaluate)
    data, pf = None, None
    if evaluate_mode:
        data = p.load(open(f'{path_data}/NASBench201/[{dataset}]_data.p', 'rb'))
        pf = p.load(open(f'{path_data}/NASBench201/[{dataset}]_pareto_front(testing).p', 'rb'))
        pf[:, 0] /= 1e3

    for run_i in range(n_runs):
        logging.info(f'Run ID: {run_i + 1}')
        path_results_ = path_results + '/' + f'{run_i}'

        try:
            os.mkdir(path_results_)
        except FileExistsError:
            pass
        logging.info(f'Path for saving results: {path_results_}')

        random_seed = random_seeds_list[run_i]
        logging.info(f'Random seed: {run_i}')
        set_seed(random_seed)

        s = time.time()
        list_prune_arch = mo_prune(tf_ind=tf_metric, path_data=path_data, path_results=path_results_,
                                   seed=random_seed)
        executed_time = time.time() - s
        logging.info(f'Prune Done! Executed time: {executed_time} seconds.\n')
        p.dump(executed_time, open(f'{path_results_}/running_time.p', 'wb'))

        if evaluate_mode:
            evaluate(list_prune_arch=list_prune_arch, data=data, pf=pf, path_results=path_results_)

        with open(f'{path_results_}/logging.txt', 'w') as f:
            f.write(f'******* PROBLEM *******\n')
            f.write(f'- Benchmark: NAS-Bench-201\n')
            f.write(f'- Dataset: CIFAR-10\n\n')

            f.write(f'******* RUNNING *******\n')
            f.write(f'- Pruning:\n')
            f.write(f'\t+ The first objective (minimize): FLOPs\n')
            f.write(f'\t+ The second objective (minimize): -Synflow\n')
            f.write(f'- Evaluate:\n')
            f.write(f'\t+ The first objective (minimize): FLOPs\n')
            f.write(f'\t+ The second objective (minimize): test error\n\n')

            f.write(f'******* ENVIRONMENT *******\n')
            f.write(f'- ID experiments: {run_i}\n')
            f.write(f'- Random seed: {random_seed}\n')
            f.write(f'- Path for saving results: {path_results_}\n\n')
        print('-' * 40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' ENVIRONMENT '''
    parser.add_argument('--path_data', type=str, default=None, help='path for loading data')
    parser.add_argument('--path_results', type=str, default=None, help='path for saving results')
    parser.add_argument('--n_runs', type=int, default=31, help='number of experiment runs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--evaluate', type=int, default=1, help='evaluate after pruning')
    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    main(args)
