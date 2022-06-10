import pickle as p
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_IGD_value, get_front_0
from nasbench import wrap_api as api
from scipy.interpolate import interp1d
from copy import deepcopy

API = api.NASBench_()

plt.rc('legend', fontsize=13)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.rc('axes', labelsize=13)
plt.rcParams["font.family"] = "serif"

from matplotlib.ticker import FormatStrFormatter


def visualize_2D(objective_0_mean, tmp_objective_1_mean, tmp_objective_1_stdev, label, line):
    color = line[0]
    style = line[1]
    ax.plot(objective_0_mean, tmp_objective_1_mean, c=color, ls=style, label=label, linewidth=2)
    ax.fill_between(objective_0_mean,
                     tmp_objective_1_mean - tmp_objective_1_stdev,
                     tmp_objective_1_mean + tmp_objective_1_stdev, alpha=0.2, fc=color)


def format_ops_matrix_raw_2(ops_matrix_raw):
    ops_matrix = ['input']
    for ops in ops_matrix_raw[1:-1]:
        ops_matrix.append(ops[-1])
    ops_matrix.append('output')
    return ops_matrix

def evaluate(edge_matrix_full, ops_matrix_full, F_full):
    idx_front_0 = get_front_0(F_full)
    edge_matrix_final = np.array(deepcopy(edge_matrix_full))[idx_front_0]
    ops_matrix_final = np.array(deepcopy(ops_matrix_full))[idx_front_0]

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
        F_lst.append([params, np.round(1 - data['108'][module_hash]['test_acc'], 6)])
        total_train_time += data['108'][module_hash]['train_time']

    F_lst = np.array(F_lst)
    adj_matrix_lst = np.array(adj_matrix_lst)
    ops_matrix_lst = np.array(ops_matrix_lst)

    idx = get_front_0(F_lst)
    F_lst = F_lst[idx]
    adj_matrix_lst = adj_matrix_lst[idx]
    ops_matrix_lst = ops_matrix_lst[idx]
    F_lst = np.unique(F_lst, axis=0)
    IGD = np.round(calculate_IGD_value(pareto_front=pf, non_dominated_front=F_lst), 6)
    return adj_matrix_lst, ops_matrix_lst, F_lst, IGD

def process_nEvals_hist(nEvals_hist):
    s_value = nEvals_hist[0]
    all_index = []
    all_value = []
    for i, value in enumerate(nEvals_hist):
        if value != s_value:
            all_index.append(i)
            all_value.append(s_value)
            s_value = value
    if s_value != all_value[-1]:
        all_index.append(-1)
        all_value.append(s_value)
    return all_index, all_value

def visualize_results_nEvals():
    labels = ['MOENAS', 'TF-MOENAS', 'TF-MOPNAS (our)']
    colors = {'TF-MOPNAS (our)': ['red', 'solid'],
              'MOENAS': ['blue', '-.'],
              'TF-MOENAS': ['green', '-.']}

    file_lst = [f'./results/101/raw_results_MOENAS.p',
                f'./results/101/raw_results_TF-MOENAS.p',
                f'./results/101/results_TF-MOPNAS.p']

    IGD_mean_algo = []
    IGD_std_algo = []

    nEvals_algo = []
    for file in file_lst:
        rs = p.load(open(file, 'rb'))
        nEvals = rs['nEvals_all'][0]
        nEvals_algo.append(nEvals)
        IGD_all = rs['IGD_all']

        # print(file)
        # print(IGD_all)
        # print(len(IGD_all))
        IGD_mean = np.mean(IGD_all, axis=0)
        IGD_std = np.std(IGD_all, axis=0)

        IGD_mean_algo.append(IGD_mean)
        IGD_std_algo.append(IGD_std)

    ends = [20, 100, 1000, 3000]
    for i in range(len(IGD_mean_algo)):
        visualize_2D(nEvals_algo[i], IGD_mean_algo[i], IGD_std_algo[i], labels[i], line=colors[labels[i]])

    ax.set_xscale('log')
    ax.set_xticks(ends)
    ax.get_xaxis().set_major_formatter(FormatStrFormatter('%2d'))
    ax.set_xlabel('#Evals')
    ax.set_ylabel('IGD')
    plt.grid(True, linestyle='--')
    plt.legend(loc=1)
    plt.title('NAS-Bench-101 (x-axis: nEvals)', fontsize=20)
    plt.savefig(f'./results/nEvals_IGD_101.jpg', bbox_inches='tight', pad_inches=0.1,
                dpi=300)
    plt.clf()
    # plt.show()

def visualize_results_hours():
    labels = ['MOENAS', 'TF-MOENAS', 'TF-MOPNAS (our)']
    colors = {'TF-MOPNAS (our)': ['red', 'solid'],
              'MOENAS': ['blue', '-.'],
              'TF-MOENAS': ['green', '-.']}

    file_lst = [f'./results/101/raw_results_MOENAS.p',
                f'./results/101/raw_results_TF-MOENAS.p',
                f'./results/101/results_TF-MOPNAS.p']

    IGD_mean_algo = []
    IGD_std_algo = []

    rt_algo = []
    nEvals_algo = []
    for file in file_lst:
        rs = p.load(open(file, 'rb'))
        nEvals = rs['nEvals_all'][0]
        nEvals_algo.append(nEvals)
        IGD_all = rs['IGD_all']

        rt_all = rs['running_time_all']
        # print(rt_all)
        IGD_mean = np.mean(IGD_all, axis=0)
        IGD_std = np.std(IGD_all, axis=0)
        rt_mean = np.round(np.mean(rt_all, axis=0) / 3600, 4)

        IGD_mean_algo.append(IGD_mean)
        IGD_std_algo.append(IGD_std)

        rt_algo.append(rt_mean)

    ends = [0.004, 0.022, 0.359, 0.661, 97.155]
    for i in range(len(IGD_mean_algo)):
        # visualize_2D(nEvals_algo[i], IGD_mean_algo[i], IGD_std_algo[i], labels[i], line=colors[labels[i]])
        #
        # ends += [rt_algo[i][0], rt_algo[i][-1]]
        visualize_2D(rt_algo[i], IGD_mean_algo[i], IGD_std_algo[i], labels[i], line=colors[labels[i]])

    ax.set_xscale('log')
    ax.set_xticks(ends)
    ax.get_xaxis().set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_xlabel('GPUs Hours')
    ax.set_ylabel('IGD')
    plt.xticks(rotation=90)

    plt.grid(True, linestyle='--')
    plt.legend(loc=1)
    # plt.legend(loc=4)
    plt.title('NAS-Bench-101 (x-axis: GPU Hours)', fontsize=20)
    plt.savefig(f'./results/runtime_IGD_101.jpg', bbox_inches='tight', pad_inches=0.1,
                dpi=300)
    plt.clf()
    # plt.show()

def summarize_rs_nsganet_synflow():
    # Can use for both nsganet and synflow
    variants_lst = ['MOENAS', 'TF-MOENAS']
    for variant in variants_lst:
        p_file = f'./results/101/{variant}'

        nEvals_all = []
        runningtime_all = []
        IGD_all = []
        EA_all = []
        for i in range(31):
            p_file_ = p_file + f'/{i}'
            try:
                EA_lst = p.load(open(p_file_ + '/E_Archive_evaluate_each_gen.p', 'rb'))
            except:
                raise FileNotFoundError('Please download the prepared results!')
            EA_new = []
            for ea in EA_lst:
                EA = {
                    'X_lst': ea[0],
                    'hashKey_lst': ea[1],
                    'F_lst': ea[-1],
                }
                EA_new.append(EA)
                F = np.array(ea[-1])
            nEvals_rt_IGD = np.array(p.load(open(p_file_ + '/#Evals_runningtime_IGD_each_gen.p', 'rb')))
            rt = np.array(nEvals_rt_IGD[:, 1])
            IGD = np.array(nEvals_rt_IGD[:, 2])
            nEvals = np.array(nEvals_rt_IGD[:, 0], dtype=int)

            EA_all.append(EA_new)
            IGD_all.append(IGD)
            runningtime_all.append(rt)
            nEvals_all.append(nEvals)

        IGD_all = np.array(IGD_all)
        rt_all = np.array(runningtime_all)
        nEvals_all = np.array(nEvals_all)
        rs = {
            'nEvals_all': nEvals_all,
            'EA_all': EA_all,
            'running_time_all': rt_all,
            'IGD_all': IGD_all,
        }
        p.dump(rs, open(f'./results/101/raw_results_{variant}.p', 'wb'))

def create_results_nsganet_synflow_prune():
    # Can use for nsganet, synflow, and prune
    for algo in ['MOENAS', 'TF-MOENAS', 'TF-MOPNAS']:
        if algo == 'TF-MOPNAS':
            RS = p.load(open(f'./results/101/results_{algo}.p', 'rb'))
        else:
            RS = p.load(open(f'./results/101/raw_results_{algo}.p', 'rb'))
        # print(RS)
        best_arch_found = []
        final_IGD = []
        final_EA = []
        final_running_time = []

        for i in range(31):
            IGD = np.round(RS['IGD_all'][i][-1], 6)
            EA = RS['EA_all'][i][-1]
            F = np.array(RS['EA_all'][i][-1]['F_lst'])
            best_arch = np.round(100.0 - np.min(F[:, 1]) * 100, 2)
            rt = RS['running_time_all'][i][-1]
            best_arch_found.append(best_arch)
            final_IGD.append(IGD)
            final_EA.append(EA)
            final_running_time.append(rt)

        rs_sum = {
            'best_arch_found': best_arch_found,
            'final_EA': final_EA,
            'final_IGD': final_IGD,
            'running_time_avg': int(np.mean(final_running_time)) + 1
        }

        p.dump(rs_sum, open(f'./results/101/{algo}_C10.p', 'wb'))

def create_raw_results_prune():
    p_file = './results/101/TF-MOPNAS'

    nEvals_all = []
    running_time_all = []
    EA_all = []
    IGD_all = []

    for i in range(31):
        p_file_ = p_file + f'/{i}'
        try:
            nEvals_hist, edge_matrix_full, ops_matrix_full, F_full = p.load(open(p_file_ + '/after_search_result_history.p', 'rb'))
        except:
            raise FileNotFoundError('Please do experiments!')
        total_running_time = p.load(open(p_file_ + '/running_time.p', 'rb'))
        IDX, nEvals_hist_new = process_nEvals_hist(nEvals_hist)
        rt_hist = (total_running_time / nEvals_hist_new[-1] * np.array(nEvals_hist_new)).tolist()
        IGD_hist = []
        EA_hist = []
        for idx in IDX:
            adj_matrix, ops_matrix, F_lst, IGD = evaluate(edge_matrix_full[:idx], ops_matrix_full[:idx], F_full[:idx])
            IGD_hist.append(IGD)
            EA = {
                'adj_matrix_lst': adj_matrix,
                'ops_matrix_lst': ops_matrix,
                'F_lst': F_lst
            }
            EA_hist.append(EA)
        nEvals_all.append(nEvals_hist_new)
        running_time_all.append(rt_hist)
        EA_all.append(EA_hist)
        IGD_all.append(IGD_hist)

    rs = {
        'nEvals_all': nEvals_all,
        'running_time_all': running_time_all,
        'EA_all': EA_all,
        'IGD_all': IGD_all,
    }
    p.dump(rs, open('./results/101/raw_results_TF-MOPNAS.p', 'wb'))

def create_results_prune():
    p_file = './results/101/raw_results_TF-MOPNAS.p'
    rs = p.load(open(p_file, 'rb'))

    nEvals_all = []
    running_time_all = []
    IGD_all = []
    new_nEvals_history = range(100, 3001, 20)

    for i in range(31):
        nEvals = rs['nEvals_all'][i]
        IGD = rs['IGD_all'][i]
        rt = rs['running_time_all'][i]
        f = interp1d(nEvals, IGD, kind='cubic')
        new_IGD_history = f(new_nEvals_history)
        IGD_all.append(new_IGD_history)

        f = interp1d(nEvals, rt)
        new_rt_history = f(new_nEvals_history)

        running_time_all.append(new_rt_history)
        nEvals_all.append(list(new_nEvals_history))

    rs_new = {
        'nEvals_all': np.array(nEvals_all, dtype=int),
        'running_time_all': np.array(running_time_all),
        'EA_all': rs['EA_all'],
        'IGD_all': np.array(IGD_all),
    }
    p.dump(rs_new, open('./results/101/results_TF-MOPNAS.p', 'wb'))

if __name__ == '__main__':
    data = p.load(open(f'./benchmark_data/NASBench101/data.p', 'rb'))
    pf = p.load(open(f'./benchmark_data/NASBench101/pareto_front(testing).p', 'rb'))

    ''' Logging results (prune) '''
    create_raw_results_prune()
    create_results_prune()

    ''' Summarize nsganet, synflow results '''
    summarize_rs_nsganet_synflow()
    create_results_nsganet_synflow_prune()

    ''' Visualize results (nEvals)'''
    fig, ax = plt.subplots()
    visualize_results_nEvals()

    ''' Visualize results (hours)'''
    fig, ax = plt.subplots()
    visualize_results_hours()