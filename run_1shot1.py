import argparse
import itertools
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
from zero_cost_methods import get_config_for_zero_cost_predictor, get_zero_cost_predictor
import matplotlib.pyplot as plt

API = api.NASBench_()
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


def mo_edges_prune(search_space, tf_ind, path_data, path_results, seed):
    logging.info('--> Edges Pruning <--')

    if search_space == 'NAS101-3':
        ops_matrix = [INPUT,
                      [CONV1X1, CONV3X3, MAXPOOL3X3],
                      [CONV1X1, CONV3X3, MAXPOOL3X3],
                      [CONV1X1, CONV3X3, MAXPOOL3X3],
                      [CONV1X1, CONV3X3, MAXPOOL3X3],
                      [CONV1X1, CONV3X3, MAXPOOL3X3],
                      OUTPUT]
        i_prune = 2
        list_parents = np.array([[[False, False, False, False, False, False, False],
                                  [True, False, False, False, False, False, False],
                                  [True, True, False, False, False, False, False],
                                  [True, True, True, False, False, False, False],
                                  [True, True, True, True, False, False, False],
                                  [True, True, True, True, True, False, False],
                                  [True, True, True, True, True, True, False]]])
    elif search_space == 'NAS101-2':
        ops_matrix = [INPUT,
                      [CONV1X1, CONV3X3, MAXPOOL3X3],
                      [CONV1X1, CONV3X3, MAXPOOL3X3],
                      [CONV1X1, CONV3X3, MAXPOOL3X3],
                      [CONV1X1, CONV3X3, MAXPOOL3X3],
                      [CONV1X1],
                      OUTPUT]
        i_prune = 2
        list_parents = np.array([[[False, False, False, False, False, False],
                                  [True, False, False, False, False, False],
                                  [True, True, False, False, False, False],
                                  [True, True, True, False, False, False],
                                  [True, True, True, True, False, False],
                                  [True, True, True, True, True, False]]])
    else:
        ops_matrix = [INPUT,
                      [CONV1X1, CONV3X3, MAXPOOL3X3],
                      [CONV1X1, CONV3X3, MAXPOOL3X3],
                      [CONV1X1, CONV3X3, MAXPOOL3X3],
                      [CONV1X1, CONV3X3, MAXPOOL3X3],
                      [CONV1X1],
                      OUTPUT]
        list_parents = np.array([[[False, False, False, False, False, False],
                                  [True, False, False, False, False, False],
                                  [True, True, False, False, False, False],
                                  [True, True, True, False, False, False],
                                  [True, True, True, True, False, False],
                                  [True, True, True, True, True, False]]])
        i_prune = 3
    max_nPrunes = len(list_parents[-1])

    config = get_config_for_zero_cost_predictor(search_space='NASBench101', dataset='CIFAR-10',
                                                seed=seed, path_data=path_data)
    ZC_predictor = get_zero_cost_predictor(config=config, method_type=tf_ind)

    id_arch = 0
    while i_prune <= max_nPrunes - 1:
        if search_space == 'NAS101-3':
            logging.info(f'Pruning Times: {i_prune - 2 + 1}\n')
        else:
            logging.info(f'Pruning Times: {i_prune - 3 + 1}\n')
        list_arch_child = []
        F_arch_child = []
        for arch in list_parents:
            arch_child = deepcopy(arch)
            arch_child[i_prune] = np.array([False for _ in range(len(arch[-1]))])
            all_combs = list(itertools.combinations(list(range(i_prune)), num_parents_per_node[search_space][f'{i_prune}']))
            for comb in all_combs:
                id_arch += 1
                arch_child[i_prune][list(comb)] = True
                list_arch_child.append(deepcopy(arch_child))
                parents = edge_matrix_2_parent_config(list_arch_child[-1])
                if search_space == 'NAS101-3':
                    adj_matrix = create_nasbench_adjacency_matrix_with_loose_ends(parents)
                else:
                    adj_matrix = upscale_to_nasbench_format(create_nasbench_adjacency_matrix_with_loose_ends(parents))
                # print(adj_matrix)

                spec = api.ModelSpec(adj_matrix, ops_matrix)

                network = Network(spec,
                                  stem_out=128,
                                  num_stacks=3,
                                  num_mods=3,
                                  num_classes=10)
                flops, params = get_model_infos(network, xshape)

                params = np.round(params/1e2, 6)
                tf_metric_value = ZC_predictor.query__(spec=spec)[tf_ind]

                F_value = [params, -tf_metric_value]
                logging.info(f'ID Arch: {id_arch}')
                logging.info(f'Edges matrix:\n{adj_matrix}')
                logging.info(f'nParams: {F_value[0]}')
                logging.info(f'Synflow: {F_value[-1]}\n')
                F_arch_child.append(F_value)
                arch_child[i_prune][list(comb)] = False

        idx_front_0 = get_front_0(F_arch_child)
        list_parents = np.array(deepcopy(list_arch_child))[idx_front_0]
        logging.info(f'Number of architectures on the next pruning time: {len(list_parents)}\n')
        i_prune += 1

    p.dump(list_parents, open(f'{path_results}/after_edges_pruning_result.p', 'wb'))
    return list_parents

def mo_ops_prune(list_edges_prunned_matrix, search_space, tf_ind, path_data, path_results, seed):
    logging.info('--> Operations Pruning <--')
    config = get_config_for_zero_cost_predictor(search_space='NASBench101', dataset='CIFAR-10',
                                                seed=seed, path_data=path_data)
    ZC_predictor = get_zero_cost_predictor(config=config, method_type=tf_ind)

    edge_matrix_full = []
    ops_matrix_full = []
    F_full = []
    id_arch = 0
    for edge_matrix_ori in list_edges_prunned_matrix:
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

        parents = edge_matrix_2_parent_config(edge_matrix_ori)
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

                    tf_metric_value = ZC_predictor.query__(spec=spec)[tf_ind]

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
            edge_matrix_full.append(ADJ_MATRIX)
            OPS_MATRIX = format_ops_matrix_raw_1(ops_matrix_raw)
            ops_matrix_full.append(OPS_MATRIX)
            F_full.append(F_parents[n])
            logging.info(f'Operations matrix:\n{OPS_MATRIX}')
        print('-'*40)
    p.dump([edge_matrix_full, ops_matrix_full, F_full], open(f'{path_results}/after_ops_pruning_result_history.p', 'wb'))
    idx_front_0 = get_front_0(F_full)
    edge_matrix_final = np.array(deepcopy(edge_matrix_full))[idx_front_0]
    ops_matrix_final = np.array(deepcopy(ops_matrix_full))[idx_front_0]
    p.dump([edge_matrix_final, ops_matrix_final], open(f'{path_results}/after_ops_pruning_result.p', 'wb'))
    # return edge_matrix_full, ops_matrix_full, F_full
    return edge_matrix_final, ops_matrix_final

def evaluate(edge_matrix_final, ops_matrix_final, search_space, data, pf, path_results):
    adj_matrix_lst = []
    ops_matrix_lst = []
    F_lst = []
    total_train_time = 0.0
    for i in range(len(edge_matrix_final)):
        ADJ_MATRIX = edge_matrix_final[i]
        OPS_MATRIX = format_ops_matrix_raw_2(ops_matrix_final[i])
        adj_matrix_lst.append(ADJ_MATRIX)
        ops_matrix_lst.append(OPS_MATRIX)
        spec = api.ModelSpec(ADJ_MATRIX, OPS_MATRIX)
        module_hash = API.get_module_hash(spec)
        params = np.round(data['108'][module_hash]['n_params'] / 1e8, 6)
        F_lst.append([params, 1 - data['108'][module_hash]['test_acc']])
        total_train_time += data['108'][module_hash]['train_time']

    F_lst = np.array(F_lst)
    adj_matrix_lst = np.array(adj_matrix_lst)
    ops_matrix_lst = np.array(ops_matrix_lst)
    idx = get_front_0(F_lst)
    F_lst = F_lst[idx]
    F_lst = np.unique(F_lst, axis=0)
    adj_matrix_lst = adj_matrix_lst[idx]
    ops_matrix_lst = ops_matrix_lst[idx]
    IGD = np.round(calculate_IGD_value(pareto_front=pf, non_dominated_front=F_lst), 6)
    logging.info(f'Evaluate Done! IGD: {IGD}')

    rs = {
        'n_archs_evaluated': len(edge_matrix_final),
        'adj_matrix_lst': adj_matrix_lst,
        'ops_matrix_lst': ops_matrix_lst,
        'F_lst': F_lst,
        'total_train_time': total_train_time,
        'IGD': IGD
    }
    p.dump(rs, open(f'{path_results}/results_(evaluation).p', 'wb'))

    plt.scatter(F_lst[:, 0], F_lst[:, 1], facecolors='blue', s=30, label='approximation front')
    plt.scatter(pf[:, 0], pf[:, 1], edgecolors='red', facecolors='none', s=60, label='pareto front')
    plt.legend()
    plt.title(f'NAS-Bench-101, {search_space}, Synflow')
    plt.savefig(f'{path_results}/approximation_front.jpg')
    plt.clf()


def main(kwargs):
    n_runs = kwargs.n_runs
    init_seed = kwargs.seed
    random_seeds_list = [init_seed + run * 100 for run in range(n_runs)]

    search_space = 'NAS101-3'
    if kwargs.path_data is None:
        path_data = './benchmark_data'
    else:
        path_data = kwargs.path_data
    if kwargs.path_results is None:
        path_results = './results/1shot1/TF-MOPNAS(both)'
    else:
        path_results = kwargs.path_results
    tf_metric = 'synflow'
    evaluate_mode = bool(kwargs.evaluate)
    data, pf = None, None
    if evaluate_mode:
        data = p.load(open(f'{path_data}/NASBench101/data.p', 'rb'))
        pf = p.load(open(f'{path_data}/NASBench101/pareto_front(testing)_{search_space}.p', 'rb'))

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
        list_edges_prunned_matrix = mo_edges_prune(tf_ind=tf_metric, search_space=search_space,
                                                   path_data=path_data, path_results=path_results_, seed=random_seed)
        e1 = time.time()
        s1 = e1 - s
        logging.info(f'Prune Edges - Done. Executed time: {s1} seconds.\n')

        edge_matrix_lst, ops_matrix_lst = mo_ops_prune(list_edges_prunned_matrix=list_edges_prunned_matrix,
                                                       tf_ind=tf_metric, search_space=search_space,
                                                       path_data=path_data, path_results=path_results_,
                                                       seed=random_seed)
        e2 = time.time()
        s2 = e2 - e1
        logging.info(f'Prune Operations - Done. Executed time: {s2} seconds.\n')

        executed_time = e2 - s
        logging.info(f'Prune - Done. Executed time: {executed_time} seconds.\n')

        p.dump(executed_time, open(f'{path_results_}/running_time.p', 'wb'))

        if evaluate_mode:
            evaluate(edge_matrix_final=edge_matrix_lst,
                     ops_matrix_final=ops_matrix_lst, search_space=search_space,
                     data=data, pf=pf, path_results=path_results_)

        with open(f'{path_results_}/logging.txt', 'w') as f:
            f.write(f'******* PROBLEM *******\n')
            f.write(f'- Benchmark: NAS-Bench-101\n')
            f.write(f'- Search space: {search_space}\n\n')

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
