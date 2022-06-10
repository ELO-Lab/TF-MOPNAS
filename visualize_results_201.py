import pickle as p
import numpy as np
from utils import get_front_0
from utils import calculate_IGD_value
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.rc('legend', fontsize=13)
plt.rc('xtick', labelsize=11)
plt.rc('ytick', labelsize=11)
plt.rc('axes', labelsize=13)
plt.rcParams["font.family"] = "serif"


def convert_keep_mask_2_arch_str(keep_mask):
    arch_str = ''
    for i in range(len(keep_mask)):
        for j in range(len(keep_mask[i])):
            if keep_mask[i][j]:
                arch_str += f'{j}'
    return arch_str


def visualize_2D(objective_0_mean, tmp_objective_1_mean, tmp_objective_1_stdev, label, line):
    color = line[0]
    style = line[1]
    ax.plot(objective_0_mean, tmp_objective_1_mean, c=color, ls=style, label=label, linewidth=2)
    ax.fill_between(objective_0_mean,
                     tmp_objective_1_mean - tmp_objective_1_stdev,
                     tmp_objective_1_mean + tmp_objective_1_stdev, alpha=0.2, fc=color)

def process_rs_prune():
    dataset_lst = ['CIFAR-100', 'ImageNet16-120']
    for dataset in dataset_lst:
        data = p.load(open(f'./benchmark_data/NASBench201/[{dataset}]_data.p', 'rb'))
        pf = p.load(open(f'./benchmark_data/NASBench201/[{dataset}]_pareto_front(testing).p', 'rb'))
        pf[:, 0] /= 1e3

        path_file = f'./results/201/TF-MOPNAS'
        IGD_lst = []

        best_acc_lst = []
        for i in range(31):
            path_file_ = path_file + f'/{i}'
            try:
                list_prune_arch = p.load(open(f'{path_file_}/after_prune_result.p', 'rb'))
            except:
                raise FileNotFoundError('Please do experiments!')
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
            if dataset == 'CIFAR-100':
                p.dump(rs, open(f'{path_file_}/results_(evaluation)_C100_T.p', 'wb'))
            else:
                p.dump(rs, open(f'{path_file_}/results_(evaluation)_IN16_T.p', 'wb'))

            best_acc = np.round((1 - np.min(final_list_F_arch[:, 1])) * 100, 2)
            best_acc_lst.append(best_acc)
            IGD_lst.append(IGD)

def summarize_rs_nsganet_synflow():
    # Can use for both nsganet and synflow
    variants_lst = ['MOENAS', 'TF-MOENAS']
    for variant in variants_lst:
        path_file = f'./results/201/{variant}'

        nEvals_all = []
        rt_all = []
        EA_all = []
        IGD_all = []

        EA_all_C100_T = []
        IGD_all_C100_T = []

        EA_all_IN16_T = []
        IGD_all_IN16_T = []

        for i in range(31):
            EA_history = []

            EA_history_C100_T = []

            EA_history_IN16_T = []
            p_file_ = path_file + f'/{i}'

            try:
                nEvals_runningtime_IGD_each_gen = p.load(open(p_file_ + '/#Evals_runningtime_IGD_evaluate_each_gen.p', 'rb'))
            except:
                raise FileNotFoundError('Please download the prepared results!')
            nEvals_runningtime_IGD_each_gen_C100_T = p.load(open(p_file_ + '/#Evals_runningtime_IGD_evaluate_C100_each_gen.p', 'rb'))
            nEvals_runningtime_IGD_each_gen_IN16_T = p.load(open(p_file_ + '/#Evals_runningtime_IGD_evaluate_IN16_each_gen.p', 'rb'))
            nEvals_runningtime_IGD_each_gen = np.array(nEvals_runningtime_IGD_each_gen)
            nEvals_runningtime_IGD_each_gen_C100_T = np.array(nEvals_runningtime_IGD_each_gen_C100_T)
            nEvals_runningtime_IGD_each_gen_IN16_T = np.array(nEvals_runningtime_IGD_each_gen_IN16_T)

            nEvals_history = (nEvals_runningtime_IGD_each_gen[:, 0].astype(int)).tolist()
            running_time_history = (nEvals_runningtime_IGD_each_gen[:, 1].astype(int) + 1).tolist()
            IGD_history = (nEvals_runningtime_IGD_each_gen[:, 2]).tolist()
            IGD_history_C100_T = (nEvals_runningtime_IGD_each_gen_C100_T[:, 2]).tolist()
            IGD_history_IN16_T = (nEvals_runningtime_IGD_each_gen_IN16_T[:, 2]).tolist()

            EA_each_gen = p.load(open(p_file_ + '/E_Archive_evaluate_each_gen.p', 'rb'))
            for EA in EA_each_gen:
                F = np.round(EA[2], 6)
                ea = {
                    'X_lst': EA[0],
                    'hashKey_lst': EA[1],
                    'F_lst': F
                }
                EA_history.append(ea)

            EA_each_gen_C100_T = p.load(open(p_file_ + '/E_Archive_evaluate_C100_each_gen.p', 'rb'))
            for EA in EA_each_gen_C100_T:
                F = np.round(EA[2], 6)
                ea = {
                    'X_lst': EA[0],
                    'hashKey_lst': EA[1],
                    'F_lst': F
                }
                EA_history_C100_T.append(ea)

            EA_each_gen_IN16_T = p.load(open(p_file_ + '/E_Archive_evaluate_IN16_each_gen.p', 'rb'))
            for EA in EA_each_gen_IN16_T:
                F = np.round(EA[2], 6)
                ea = {
                    'X_lst': EA[0],
                    'hashKey_lst': EA[1],
                    'F_lst': F
                }
                EA_history_IN16_T.append(ea)

            nEvals_all.append(nEvals_history)
            rt_all.append(running_time_history)
            EA_all.append(EA_history)
            IGD_all.append(IGD_history)

            EA_all_C100_T.append(EA_history_C100_T)
            IGD_all_C100_T.append(IGD_history_C100_T)

            EA_all_IN16_T.append(EA_history_IN16_T)
            IGD_all_IN16_T.append(IGD_history_IN16_T)

        nEvals_all = np.array(nEvals_all)
        rt_all = np.array(rt_all)
        IGD_all = np.array(IGD_all)
        IGD_all_C100_T = np.array(IGD_all_C100_T)
        IGD_all_IN16_T = np.array(IGD_all_IN16_T)

        rs_all = {
            'nEvals_all': nEvals_all,
            'running_time_all': rt_all,
            'EA_all': EA_all,
            'IGD_all': IGD_all,
        }

        rs_all_C100_T = {
            'nEvals_all': nEvals_all,
            'running_time_all': rt_all,
            'EA_all': EA_all_C100_T,
            'IGD_all': IGD_all_C100_T,
        }

        rs_all_IN16_T = {
            'nEvals_all': nEvals_all,
            'running_time_all': rt_all,
            'EA_all': EA_all_IN16_T,
            'IGD_all': IGD_all_IN16_T,
        }

        p.dump(rs_all, open(f'./results/201/raw_results_{variant}_C10.p', 'wb'))
        p.dump(rs_all_C100_T, open(f'./results/201/raw_results_{variant}_C100_T.p', 'wb'))
        p.dump(rs_all_IN16_T, open(f'./results/201/raw_results_{variant}_IN16_T.p', 'wb'))

def summarize_rs_prune():
    path_file = './results/201/TF-MOPNAS'

    best_arch_found = []
    final_EA = []
    final_IGD = []

    best_arch_found_C100_T = []
    final_EA_C100_T = []
    final_IGD_C100_T = []

    best_arch_found_IN16_T = []
    final_EA_IN16_T = []
    final_IGD_IN16_T = []

    running_time_avg = 427

    for i in range(31):
        path_file_ = path_file + f'/{i}'
        rs = p.load(open(path_file_ + '/results_(evaluation).p', 'rb'))
        best_arch = np.round(100.0 - np.min(rs['F_lst'][:, 1]) * 100, 2)
        ea = {
            'hashKey_lst': rs['arch_lst'],
            'F_lst': rs['F_lst']
        }
        best_arch_found.append(best_arch)
        final_EA.append(ea)
        final_IGD.append(rs['IGD'])

        rs_C100_T = p.load(open(path_file_ + '/results_(evaluation)_C100_T.p', 'rb'))
        best_arch = np.round(100.0 - np.min(rs_C100_T['F_lst'][:, 1]) * 100, 2)
        ea = {
            'hashKey_lst': rs_C100_T['arch_lst'],
            'F_lst': rs_C100_T['F_lst']
        }
        best_arch_found_C100_T.append(best_arch)
        final_EA_C100_T.append(ea)
        final_IGD_C100_T.append(rs_C100_T['IGD'])

        rs_IN16_T = p.load(open(path_file_ + './results_(evaluation)_IN16_T.p', 'rb'))

        best_arch = np.round(100.0 - np.min(rs_IN16_T['F_lst'][:, 1]) * 100, 2)
        ea = {
            'hashKey_lst': rs_IN16_T['arch_lst'],
            'F_lst': rs_IN16_T['F_lst']
        }
        best_arch_found_IN16_T.append(best_arch)
        final_EA_IN16_T.append(ea)
        final_IGD_IN16_T.append(rs_IN16_T['IGD'])

    rs_all = {
        'best_arch_found': best_arch_found,
        'final_EA': final_EA,
        'final_IGD': final_IGD,
        'running_time_avg': running_time_avg
    }
    p.dump(rs_all, open(f'./results/201/TF-MOPNAS_C10.p', 'wb'))

    rs_all_C100_T = {
        'best_arch_found': best_arch_found_C100_T,
        'final_EA': final_EA_C100_T,
        'final_IGD': final_IGD_C100_T,
        'running_time_avg': running_time_avg
    }
    p.dump(rs_all_C100_T, open(f'./results/201/TF-MOPNAS_C100_T.p', 'wb'))

    rs_all_IN16_T = {
        'best_arch_found': best_arch_found_IN16_T,
        'final_EA': final_EA_IN16_T,
        'final_IGD': final_IGD_IN16_T,
        'running_time_avg': running_time_avg
    }
    p.dump(rs_all_IN16_T, open(f'./results/201/TF-MOPNAS_IN16_T.p', 'wb'))

def process_rs_nsganet_synflow():
    # Can use for both nsganet and synflow
    variants_lst = ['MOENAS', 'TF-MOENAS']
    for variant in variants_lst:
        rs_C10 = p.load(open(f'./results/201/raw_results_{variant}_C10.p', 'rb'))
        rs_C100_T = p.load(open(f'./results/201/raw_results_{variant}_C100_T.p', 'rb'))
        rs_IN16_T = p.load(open(f'./results/201/raw_results_{variant}_IN16_T.p', 'rb'))

        best_arch_found = []
        final_EA = []
        final_IGD = []
        final_running_time = []

        best_arch_found_C100_T = []
        final_EA_C100_T = []
        final_IGD_C100_T = []

        best_arch_found_IN16_T = []
        final_EA_IN16_T = []
        final_IGD_IN16_T = []

        for i in range(31):
            EA = rs_C10['EA_all'][i][-1]
            F = EA['F_lst']
            best_arch = np.round(100.0 - np.min(F[:, 1]) * 100, 2)
            IGD = rs_C10['IGD_all'][i][-1]
            rt = rs_C10['running_time_all'][i][-1]

            best_arch_found.append(best_arch)
            final_IGD.append(IGD)
            final_running_time.append(rt)
            final_EA.append(EA)

            EA = rs_C100_T['EA_all'][i][-1]
            F = EA['F_lst']
            best_arch = np.round(100.0 - np.min(F[:, 1]) * 100, 2)
            IGD = rs_C100_T['IGD_all'][i][-1]

            best_arch_found_C100_T.append(best_arch)
            final_IGD_C100_T.append(IGD)
            final_EA_C100_T.append(EA)

            EA = rs_IN16_T['EA_all'][i][-1]
            F = EA['F_lst']
            best_arch = np.round(100.0 - np.min(F[:, 1]) * 100, 2)
            IGD = rs_IN16_T['IGD_all'][i][-1]

            best_arch_found_IN16_T.append(best_arch)
            final_IGD_IN16_T.append(IGD)
            final_EA_IN16_T.append(EA)

        rs_final = {
            'best_arch_found': best_arch_found,
            'final_EA': final_EA,
            'final_IGD': final_IGD,
            'running_time_avg': int(np.mean(final_running_time)) + 1
        }
        p.dump(rs_final, open(f'./results/201/{variant}_C10.p', 'wb'))

        rs_final_C100_T = {
            'best_arch_found': best_arch_found_C100_T,
            'final_EA': final_EA_C100_T,
            'final_IGD': final_IGD_C100_T,
            'running_time_avg': int(np.mean(final_running_time)) + 1
        }
        p.dump(rs_final_C100_T, open(f'./results/201/{variant}_C100_T.p', 'wb'))

        rs_final_IN16_T = {
            'best_arch_found': best_arch_found_IN16_T,
            'final_EA': final_EA_IN16_T,
            'final_IGD': final_IGD_IN16_T,
            'running_time_avg': int(np.mean(final_running_time)) + 1
        }
        p.dump(rs_final_IN16_T, open(f'./results/201/{variant}_IN16_T.p', 'wb'))


def visualize_results_hours():
    labels = ['MOENAS', 'TF-MOENAS']
    colors = {'TF-MOPNAS (our)': ['red', 'solid'],
             'MOENAS': ['blue', '-.'],
             'TF-MOENAS': ['green', '-.']}

    dataset = 'C10'
    file_prune = f'./results/201/TF-MOPNAS_{dataset}.p'
    rs_prune = p.load(open(file_prune, 'rb'))
    IGD_prune = rs_prune['final_IGD']
    IGD_mean_prune, IGD_std_prune = np.mean(IGD_prune), np.std(IGD_prune)
    rt_prune = np.round(rs_prune['running_time_avg']/3600, 4)

    file_lst = [f'./results/201/raw_results_MOENAS_{dataset}.p',
                f'./results/201/raw_results_TF-MOENAS_{dataset}.p']

    IGD_mean_algo = []
    IGD_std_algo = []

    rt_algo = []
    for file in file_lst:
        rs = p.load(open(file, 'rb'))
        IGD_all = rs['IGD_all']
        rt_all = rs['running_time_all']
        IGD_mean = np.mean(IGD_all, axis=0)
        IGD_std = np.std(IGD_all, axis=0)
        rt_mean = np.round(np.mean(rt_all, axis=0)/3600, 4)

        IGD_mean_algo.append(IGD_mean)
        IGD_std_algo.append(IGD_std)

        rt_algo.append(rt_mean)

    ends = [rt_prune]
    for i in range(len(IGD_mean_algo)):
        ends += [rt_algo[i][0], rt_algo[i][-1]]
        visualize_2D(rt_algo[i], IGD_mean_algo[i], IGD_std_algo[i], labels[i], line=colors[labels[i]])

    ax.errorbar(rt_prune, IGD_mean_prune, IGD_std_prune, label='TF-MOPNAS (our)', c='red', lw=2, fmt='o')

    ax.set_xscale('log')
    ax.set_xticks(ends)
    ax.get_xaxis().set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('GPUs Hours')
    ax.set_ylabel('IGD')
    plt.xticks(rotation=90)

    plt.grid(True, linestyle='--')
    plt.title('NASBench201', fontsize=20)
    plt.legend(loc=1)
    plt.savefig(f'./results/runtime_IGD_201_{dataset}.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)

if __name__ == '__main__':
    ''' Summarize results '''
    process_rs_prune()
    summarize_rs_prune()

    summarize_rs_nsganet_synflow()
    process_rs_nsganet_synflow()
    ''' Visualize result '''
    fig, ax = plt.subplots()
    visualize_results_hours()
