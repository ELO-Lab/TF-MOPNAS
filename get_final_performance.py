import pickle as p
import numpy as np

def evaluate_TF_MOPNAS_201():
    print('************************** NAS-Bench-201 **************************')
    path_file = f'./results/201/TF-MOPNAS'

    for dataset in ['C10', 'C100_T', 'IN16_T']:
        if dataset == 'C10':
            print('CIFAR-10')
        elif dataset == 'C100_T':
            print('CIFAR-100 (transfer)')
        else:
            print('ImageNet16-120 (transfer)')
        best_acc_lst = []
        IGD_lst = []
        rt_total = 0.0
        for i in range(31):
            path_file_ = path_file + f'/{i}'
            if dataset != 'C10':
                try:
                    rs = p.load(open(f'{path_file_}/results_(evaluation)_{dataset}.p', 'rb'))
                except:
                    raise FileNotFoundError('Not found results file! Please running "visualize_results_201.py" firstly!')
            else:
                try:
                    rs = p.load(open(f'{path_file_}/results_(evaluation).p', 'rb'))
                except:
                    raise FileNotFoundError('Not found results file! Please conducting experiments!')
            best_acc = np.round((1 - np.min(rs['F_lst'][:, 1])) * 100, 2)
            best_acc_lst.append(best_acc)

            IGD = rs['IGD']
            IGD_lst.append(IGD)
            rt = p.load(open(f'{path_file_}/running_time.p', 'rb'))
            rt_total += rt

        if dataset == 'C10':
            print(f'Total running time (average over 31 runs): {np.round(rt_total/31)}')
        IGD_mean_std = [np.mean(IGD_lst), np.std(IGD_lst)]
        IGD_mean_std = np.round(IGD_mean_std, 3)
        print(f'IGD (average over 31 runs): {IGD_mean_std[0]} ({IGD_mean_std[1]})')

        mean_std = [np.mean(best_acc_lst), np.std(best_acc_lst)]
        mean_std = np.round(mean_std, 2)
        print(f'Best architecture found (average 31 runs): {mean_std[0]} ({mean_std[1]})')
        print()

def evaluate_TF_MOPNAS_101():
    print('************************** NAS-Bench-101 **************************')
    root_path = './results/101/TF-MOPNAS'

    IGD_lst = []
    best_arch_lst = []
    total_rt = 0.0
    for i in range(31):
        try:
            rt = p.load(open(root_path + f'/{i}/running_time.p', 'rb'))
            total_rt += rt

            path_file = root_path + f'/{i}/results_(evaluation).p'
            rs = p.load(open(path_file, 'rb'))
            IGD_lst.append(rs['IGD'])
            F = rs['F_lst']
            best_arch_lst.append(np.round((1 - np.min(F[:, 1])) * 100, 2))
        except:
            raise FileNotFoundError('Not found results file! Please conducting experiments!')

    print(f'Total running time (average over 31 runs): {np.round(total_rt/ 31)}')
    mean_std = [np.mean(IGD_lst), np.std(IGD_lst)]
    mean_std = np.round(mean_std, 3)
    print(f'IGD (average over 31 runs): {mean_std[0]} ({mean_std[1]})')

    mean_std = [np.mean(best_arch_lst), np.std(best_arch_lst)]
    mean_std = np.round(mean_std, 2)
    print(f'Best architecture found (average 31 runs): {mean_std[0]} ({mean_std[1]})')
    print()

def evaluate_TF_MOPNAS_1shot1():
    print('************************** NAS-Bench-1shot1 **************************')
    for variant in ['TF-MOPNAS(both)', 'TF-MOPNAS(random)']:
        print(variant)
        root_path = f'./results/1shot1/{variant}'

        IGD_lst = []
        best_arch_lst = []
        total_rt = 0.0
        for i in range(31):
            try:
                rt = p.load(open(root_path + f'/{i}/running_time.p', 'rb'))
                total_rt += rt

                path_file = root_path + f'/{i}/results_(evaluation).p'
                rs = p.load(open(path_file, 'rb'))
                IGD_lst.append(rs['IGD'])
                F = rs['F_lst']
                best_arch_lst.append(np.round((1 - np.min(F[:, 1])) * 100, 2))
            except:
                raise FileNotFoundError('Not found results file! Please conducting experiments!')

        print(f'Total running time (average over 31 runs): {np.round(total_rt/ 31)}')
        mean_std = [np.mean(IGD_lst), np.std(IGD_lst)]
        mean_std = np.round(mean_std, 3)
        print(f'IGD (average over 31 runs): {mean_std[0]} ({mean_std[1]})')
        print()


if __name__ == '__main__':
    evaluate_TF_MOPNAS_101()
    evaluate_TF_MOPNAS_1shot1()
    evaluate_TF_MOPNAS_201()
