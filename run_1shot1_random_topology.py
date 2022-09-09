import argparse
import logging
import os
import pickle as p
import sys
import time
from copy import deepcopy

from utils import get_model_infos
from pruning_models.model_101 import *
from nasbench import wrap_api as api
from utils import get_front_0
from utils import calculate_IGD_value
from utils import set_seed
from training_free_metrics import get_config_for_training_free_calculator, get_training_free_calculator
import matplotlib.pyplot as plt

xshape = (1, 3, 32, 32)
INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

num_parents_per_node = {
    'NAS101-1': {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 2,
        '4': 2,
        '5': 2
    },
    'NAS101-2': {
        '0': 0,
        '1': 1,
        '2': 1,
        '3': 2,
        '4': 2,
        '5': 3
    },
    'NAS101-3': {
        '0': 0,
        '1': 1,
        '2': 1,
        '3': 1,
        '4': 2,
        '5': 2,
        '6': 2
    },
}

OPS_OPTIONS = np.array([CONV1X1, CONV3X3, MAXPOOL3X3])

def format_ops_matrix_raw_2(ops_matrix_raw):
    ops_matrix = ['input']
    for ops in ops_matrix_raw[1:-1]:
        ops_matrix.append(ops[-1])
    ops_matrix.append('output')
    return ops_matrix

def upscale_to_nasbench_format(adjacency_matrix):
    """
    The search space uses only 4 intermediate nodes, rather than 5 as used in nasbench
    This method adds a dummy node to the graph which is never used to be compatible with nasbench.
    :param adjacency_matrix:
    :return:
    """
    return np.insert(
        np.insert(adjacency_matrix, 5, [0, 0, 0, 0, 0, 0], axis=1),
        5, [0, 0, 0, 0, 0, 0, 0], axis=0)


def edge_matrix_2_parent_config(edge_matrix):
    parents = {}
    for i in range(len(edge_matrix)):
        linked_nodes = []
        for j in range(len(edge_matrix[i])):
            if edge_matrix[i][j]:
                linked_nodes.append(j)
        parents[f'{i}'] = linked_nodes
    return parents

def format_ops_matrix_raw_1(ops_matrix_raw):
    ops_matrix = []
    for row in ops_matrix_raw:
        op = OPS_OPTIONS[row]
        ops_matrix.append(op.tolist())
    ops_matrix.insert(0, INPUT)
    ops_matrix.append(OUTPUT)
    return ops_matrix

def create_nasbench_adjacency_matrix_with_loose_ends(parents):
    adjacency_matrix = np.zeros([len(parents), len(parents)], dtype=int)
    for node, node_parents in parents.items():
        for parent in node_parents:
            adjacency_matrix[parent, int(node)] = 1
    return adjacency_matrix

def evaluate(final_opt_edge_matrices, final_opt_ops_matrices, api_benchmark, pf, path_results, search_space):
    adj_matrix_lst = []
    ops_matrix_lst = []
    approximation_front = []
    total_pos_training_time = 0.0
    for i in range(len(final_opt_edge_matrices)):
        ADJ_MATRIX = final_opt_edge_matrices[i]
        OPS_MATRIX = format_ops_matrix_raw_2(final_opt_ops_matrices[i])
        adj_matrix_lst.append(ADJ_MATRIX)
        ops_matrix_lst.append(OPS_MATRIX)
        spec = api.ModelSpec(ADJ_MATRIX, OPS_MATRIX)

        info = api_benchmark.query(spec)

        params = np.round(info['n_params'] / 1e8, 6)
        approximation_front.append([params, 1 - info['test_acc']])

        total_pos_training_time += info['train_time']

    approximation_front = np.array(approximation_front)
    adj_matrix_lst = np.array(adj_matrix_lst)
    ops_matrix_lst = np.array(ops_matrix_lst)
    idx = get_front_0(approximation_front)
    approximation_front = approximation_front[idx]
    approximation_front = np.unique(approximation_front, axis=0)
    adj_matrix_lst = adj_matrix_lst[idx]
    ops_matrix_lst = ops_matrix_lst[idx]
    IGD = np.round(calculate_IGD_value(pareto_front=pf, non_dominated_front=approximation_front), 6)
    logging.info(f'Evaluate -> Done!\n')
    logging.info(f'IGD: {IGD}')

    rs = {
        'n_archs_evaluated': len(final_opt_edge_matrices),
        'adj_matrix_lst': adj_matrix_lst,
        'ops_matrix_lst': ops_matrix_lst,
        'approximation_front': approximation_front,
        'total_pos_training_time': total_pos_training_time,
        'IGD': IGD,
    }
    p.dump(rs, open(f'{path_results}/results_evaluation.p', 'wb'))

    plt.scatter(approximation_front[:, 0], approximation_front[:, 1], facecolors='blue', s=30, label='approximation front')
    plt.scatter(pf[:, 0], pf[:, 1], edgecolors='red', facecolors='none', s=60, label='pareto front')
    plt.legend()
    plt.title(f'NAS-Bench-101, {search_space}, Synflow')
    plt.savefig(f'{path_results}/approximation_front.jpg')
    plt.clf()
    return IGD

def random_edges_matrix(search_space):
    while True:
        if search_space == 'NAS101-3':
            edges_matrix = np.array([[False, False, False, False, False, False, False],
                                     [True, False, False, False, False, False, False],
                                     [False, False, False, False, False, False, False],
                                     [False, False, False, False, False, False, False],
                                     [False, False, False, False, False, False, False],
                                     [False, False, False, False, False, False, False],
                                     [False, False, False, False, False, False, False]])
            idx_start = 2
        elif search_space == 'NAS101-2':
            edges_matrix = np.array([[[False, False, False, False, False, False],
                                      [True, False, False, False, False, False],
                                      [False, False, False, False, False, False],
                                      [False, False, False, False, False, False],
                                      [False, False, False, False, False, False],
                                      [False, False, False, False, False, False]]])
            idx_start = 2
        else:
            edges_matrix = np.array([[[False, False, False, False, False, False],
                                      [True, False, False, False, False, False],
                                      [True, True, False, False, False, False],
                                      [False, False, False, False, False, False],
                                      [False, False, False, False, False, False],
                                      [False, False, False, False, False, False]]])
            idx_start = 3
        idx_end = len(edges_matrix[-1])

        for i in range(idx_start, idx_end):
            n_parents = num_parents_per_node[search_space][f'{i}']
            idx = np.random.choice(range(i), n_parents, replace=False)
            edges_matrix[i][idx] = True
        return edges_matrix

def ops_prune(search_space, maxEvals, tf_ind, path_data, path_results, seed):
    logging.info('--> Operations Pruning <--')
    id_arch = 0
    maxEvals = maxEvals
    config = get_config_for_training_free_calculator(search_space='NASBench101', dataset='CIFAR-10',
                                                     seed=seed, path_data=path_data)
    tf_calculator = get_training_free_calculator(config=config, method_type=tf_ind)

    nEvals_hist = []
    edge_matrix_full = []
    ops_matrix_full = []
    F_full = []
    while id_arch < maxEvals:
        if search_space == 'NAS101-3':
            list_parents = np.array([[[True, True, True],
                                      [True, True, True],
                                      [True, True, True],
                                      [True, True, True],
                                      [True, True, True]]])
            max_nPrunes = 5
        else:
            list_parents = np.array([[[True, True, True],
                                      [True, True, True],
                                      [True, True, True],
                                      [True, True, True],
                                      [True, False, False]]])
            max_nPrunes = 4
        i_prune = 0

        edge_matrix = random_edges_matrix(search_space)
        parents = edge_matrix_2_parent_config(edge_matrix)
        if search_space == 'NAS101-3':
            ADJ_MATRIX = create_nasbench_adjacency_matrix_with_loose_ends(parents)
        else:
            ADJ_MATRIX = upscale_to_nasbench_format(create_nasbench_adjacency_matrix_with_loose_ends(parents))

        F_parents = None
        while i_prune <= max_nPrunes - 1:
            list_arch_child = []
            F_arch_child = []
            for arch in list_parents:
                arch_child = deepcopy(arch)
                arch_child[i_prune] = np.array([False, False, False])
                for j in range(len(arch_child[i_prune])):
                    id_arch += 1
                    arch_child[i_prune][j] = True
                    list_arch_child.append(deepcopy(arch_child))
                    OPS_MATRIX = format_ops_matrix_raw_1(arch_child)
                    spec = api.ModelSpec(ADJ_MATRIX, OPS_MATRIX)
                    network = Network(spec,
                                      stem_out=128,
                                      num_stacks=3,
                                      num_mods=3,
                                      num_classes=10
                                      )
                    flop, params = get_model_infos(network, xshape)
                    params = np.round(params/1e2, 6)

                    tf_metric_value = tf_calculator.compute(spec=spec)[tf_ind]

                    F_value = [params, -tf_metric_value]
                    logging.info(f'ID Arch: {id_arch}')
                    logging.info(f'Operations matrix:\n{OPS_MATRIX}')
                    logging.info(f'nParams: {F_value[0]}')
                    logging.info(f'Synflow: {F_value[-1]}\n')
                    F_arch_child.append(F_value)
                    arch_child[i_prune][j] = False
            idx_front_0 = get_front_0(F_arch_child)
            list_parents = np.array(deepcopy(list_arch_child))[idx_front_0]
            F_parents = np.array(deepcopy(F_arch_child))[idx_front_0]
            logging.info(f'Number of architectures on the next pruning time: {len(list_parents)}\n')
            i_prune += 1
        logging.info(f'Edge matrix:\n{ADJ_MATRIX}')
        for n, ops_matrix_raw in enumerate(list_parents):
            nEvals_hist.append(id_arch)
            edge_matrix_full.append(ADJ_MATRIX)
            OPS_MATRIX = format_ops_matrix_raw_1(ops_matrix_raw)
            ops_matrix_full.append(OPS_MATRIX)
            F_full.append(F_parents[n])
            logging.info(f'Operations matrix:\n{OPS_MATRIX}')
        logging.info('-'*40)
    p.dump([nEvals_hist, edge_matrix_full, ops_matrix_full, F_full], open(f'{path_results}pruning_results_history.p', 'wb'))
    idx_front_0 = get_front_0(F_full)
    edge_matrix_final = np.array(deepcopy(edge_matrix_full))[idx_front_0]
    ops_matrix_final = np.array(deepcopy(ops_matrix_full))[idx_front_0]
    p.dump([nEvals_hist, edge_matrix_final, ops_matrix_final], open(f'{path_results}/pruning_results.p', 'wb'))
    return edge_matrix_final, ops_matrix_final


def main(kwargs):
    n_runs = kwargs.n_runs
    maxEvals = kwargs.maxEvals
    init_seed = kwargs.seed
    random_seeds_list = [init_seed + run * 100 for run in range(n_runs)]

    search_space = 'NAS101-3'
    if kwargs.path_data is None:
        path_data = './benchmark_data'
    else:
        path_data = kwargs.path_data
    if kwargs.path_results is None:
        path_results = './results/1shot1/TF-MOPNAS(random)'
    else:
        path_results = kwargs.path_results
    tf_metric = 'synflow'

    API = api.NASBench_(f'{path_data}/NASBench101/data.p')
    pareto_opt_front = p.load(open(f'{path_data}/NASBench101/pareto_front(testing)_{search_space}.p', 'rb'))

    logging.info(f'******* PROBLEM *******')
    logging.info(f'- Benchmark: NAS-Bench-101')
    logging.info(f'- Dataset: CIFAR-10')
    logging.info(f'- Search space: {search_space}\n')

    logging.info(f'******* RUNNING *******')
    logging.info(f'- Pruning:')
    logging.info(f'\t+ The first objective (minimize): #params')
    logging.info(f'\t+ The second objective (minimize): -Synflow')

    logging.info(f'- Evaluate:')
    logging.info(f'\t+ The first objective (minimize): #params')
    logging.info(f'\t+ The second objective (minimize): test error\n')

    logging.info(f'******* ENVIRONMENT *******')
    logging.info(f'- Path for saving results: {path_results}\n')

    final_IGD_lst = []
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
        final_opt_edge_matrices, final_opt_ops_matrices = ops_prune(tf_ind=tf_metric, search_space=search_space,
                                                                    maxEvals=maxEvals,
                                                                    path_data=path_data, path_results=sub_path_results,
                                                                    seed=random_seed)
        e = time.time()
        executed_time = e - s
        logging.info(f'Prune - Done. Executed time: {executed_time} seconds.\n')

        p.dump(executed_time, open(f'{sub_path_results}/running_time.p', 'wb'))

        IGD = evaluate(final_opt_edge_matrices=final_opt_edge_matrices,
                       final_opt_ops_matrices=final_opt_ops_matrices, search_space=search_space,
                       api_benchmark=API, pf=pareto_opt_front, path_results=sub_path_results)
        final_IGD_lst.append(IGD)

    logging.info(f'Average IGD: {np.round(np.mean(final_IGD_lst), 4)} ({np.round(np.std(final_IGD_lst), 4)})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' ENVIRONMENT '''
    parser.add_argument('--path_data', type=str, default=None, help='path for loading data')
    parser.add_argument('--path_results', type=str, default=None, help='path for saving results')
    parser.add_argument('--maxEvals', type=int, default=3000, help='maximum number of evaluations')
    parser.add_argument('--n_runs', type=int, default=31, help='number of experiment runs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    main(args)
