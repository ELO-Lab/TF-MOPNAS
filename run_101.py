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
from zero_cost_methods import get_config_for_zero_cost_predictor, get_zero_cost_predictor
import matplotlib.pyplot as plt

API = api.NASBench_()
xshape = (1, 3, 32, 32)
INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

OPS_OPTIONS = np.array([CONV1X1, CONV3X3, MAXPOOL3X3])

def format_ops_matrix_raw_2(ops_matrix_raw):
    ops_matrix = ['input']
    for ops in ops_matrix_raw[1:-1]:
        ops_matrix.append(ops[-1])
    ops_matrix.append('output')
    return ops_matrix

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


def random_edges_matrix():
    OPS_MATRIX = [INPUT, CONV1X1, CONV1X1, MAXPOOL3X3, MAXPOOL3X3, CONV3X3, OUTPUT]
    while True:
        edges_matrix = np.array([[False, False, False, False, False, False, False],
                               [False, False, False, False, False, False, False],
                               [False, False, False, False, False, False, False],
                               [False, False, False, False, False, False, False],
                               [False, False, False, False, False, False, False],
                               [False, False, False, False, False, False, False],
                               [False, False, False, False, False, False, False]])
        lst_idx = list(range(1, 7))
        remaining_edges = 9
        while len(lst_idx) > 0:
            idx = np.random.choice(lst_idx)
            lst_idx.remove(idx)
            IDX = np.random.randint(0, 2, idx, dtype=bool)
            idx_edges = np.argwhere(IDX == True)
            if len(idx_edges) == 0:
                pass
            else:
                sum_edges = len(idx_edges[-1])
                if sum_edges > remaining_edges:
                    idx_ = np.random.choice(idx_edges[-1], sum_edges - remaining_edges, replace=False)
                    IDX[idx_] = False
                    break
            edges_matrix[idx, :idx] = IDX
        parents = edge_matrix_2_parent_config(edges_matrix)
        ADJ_MATRIX = create_nasbench_adjacency_matrix_with_loose_ends(parents)
        model_spec = api.ModelSpec(ADJ_MATRIX, OPS_MATRIX)
        if API.is_valid(model_spec):
            return edges_matrix

def mo_prune(tf_ind, path_data, path_results, seed):
    id_arch = 0
    maxEvals = 3000
    config = get_config_for_zero_cost_predictor(search_space='NASBench101', dataset='CIFAR-10',
                                                seed=seed, path_data=path_data)
    ZC_predictor = get_zero_cost_predictor(config=config, method_type=tf_ind)
    nEvals_hist = []
    edge_matrix_full = []
    ops_matrix_full = []
    F_full = []
    while id_arch < maxEvals:
        list_parents = np.array([[[True, True, True],
                                  [True, True, True],
                                  [True, True, True],
                                  [True, True, True],
                                  [True, True, True]]])
        max_nPrunes = 5
        i_prune = 0

        edge_matrix = random_edges_matrix()
        parents = edge_matrix_2_parent_config(edge_matrix)
        ADJ_MATRIX = create_nasbench_adjacency_matrix_with_loose_ends(parents)

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
                    params = np.round(params / 1e2, 6)

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
            nEvals_hist.append(id_arch)
            edge_matrix_full.append(ADJ_MATRIX)
            OPS_MATRIX = format_ops_matrix_raw_1(ops_matrix_raw)
            ops_matrix_full.append(OPS_MATRIX)
            F_full.append(F_parents[n])
            logging.info(f'Operations matrix:\n{OPS_MATRIX}')
        print('-' * 40)
    p.dump([nEvals_hist, edge_matrix_full, ops_matrix_full, F_full], open(f'{path_results}/after_search_result_history.p', 'wb'))
    idx_front_0 = get_front_0(F_full)
    edge_matrix_final = np.array(deepcopy(edge_matrix_full))[idx_front_0]
    ops_matrix_final = np.array(deepcopy(ops_matrix_full))[idx_front_0]
    p.dump([nEvals_hist, edge_matrix_final, ops_matrix_final], open(f'{path_results}/after_search_result.p', 'wb'))
    return edge_matrix_final, ops_matrix_final


def evaluate(edge_matrix_final, ops_matrix_final, data, pf, path_results):
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
    plt.title(f'NAS-Bench-101, Synflow')
    plt.savefig(f'{path_results}/approximation_front.jpg')
    plt.clf()


def main(kwargs):
    n_runs = kwargs.n_runs
    init_seed = kwargs.seed
    random_seeds_list = [init_seed + run * 100 for run in range(n_runs)]

    if kwargs.path_data is None:
        path_data = './benchmark_data'
    else:
        path_data = kwargs.path_data
    if kwargs.path_results is None:
        path_results = './results/101/TF-MOPNAS'
    else:
        path_results = kwargs.path_results
    tf_metric = 'synflow'
    evaluate_mode = bool(kwargs.evaluate)
    data, pf = None, None
    if evaluate_mode:
        data = p.load(open(f'{path_data}/NASBench101/data.p', 'rb'))
        pf = p.load(open(f'{path_data}/NASBench101/pareto_front(testing).p', 'rb'))

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
        edge_matrix_lst, ops_matrix_lst = mo_prune(tf_ind=tf_metric,
                                                   path_data=path_data, path_results=path_results_,
                                                   seed=random_seed)
        e = time.time()
        executed_time = e - s
        logging.info(f'Search - Done. Executed time: {executed_time} seconds.\n')

        p.dump(executed_time, open(f'{path_results_}/running_time.p', 'wb'))

        if evaluate_mode:
            evaluate(edge_matrix_final=edge_matrix_lst,
                     ops_matrix_final=ops_matrix_lst,
                     data=data, pf=pf, path_results=path_results_)

        with open(f'{path_results_}/logging.txt', 'w') as f:
            f.write(f'******* PROBLEM *******\n')
            f.write(f'- Benchmark: NAS-Bench-101\n\n')

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
