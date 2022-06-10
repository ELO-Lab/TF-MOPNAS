# TF-MOPNAS: Training-free multi-objective pruning-based neural architecture search
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE.md)

Quan Minh Phan, Ngoc Hoang Luong.

In ICCCI 2022.
## Setup
- Clone this repo.
- Install necessary packages.
```
$ pip install -r requirements.txt
```
- If you meet a problem involving to ```fvcore``` package, please re-install it.
```shell
$ pip install fvcore==0.1.5post20220119
```
- Although we utilize NAS-Bench-101, NAS-Bench-1shot1, and NAS-Bench-201 benchmarks for our experiments, we do not employ directly their APIs to reduce the time and memory capability (for loading their APIs).
In our experiments, we downloaded the database, modified their APIs, and used on our way.
To reproduce the results in the paper (or conducting new experiments by yourself), we thus have to download the data at first. Download data in [here](https://drive.google.com/drive/folders/1jAX-By0UUOld_vLRLBLX1GppQ6lhcOvS?usp=sharing), unzip and put all folders into ```benchmark_data``` folder.
<!-- You can compare our data to the benchmarks' APIs by following the instructions and running code in .... -->
## Usage
### 1. Search
You can reproduce all experimental results by **re-conducting all experiments by yourself** or by **downloading**. 
#### 1.1. Do experiments from scratch
**TF-MOPNAS**
```shell
$ python <running_file> --n_runs <the_number_of_experiment_runs, default: 31> --seed <initial_random_seed, default: 0>
```
To conduct searches on a specific search space, we just run the corresponding file.
|running_file               |Benchmark              |  Variant                               |              
|:--------------------------|:----------------------|:---------------------------------------|
|`mo_prune_101.py`          |NAS-Bench-101          |TF-MOPNAS (random cell topologies + edges pruning)|
|`mo_prune_1shot1.py`       |NAS-Bench-1shot1       |TF-MOPNAS (cell topologies + edges pruning)|
|`mo_prune_1shot1_random.py`|NAS-Bench-1shot1       |TF-MOPNAS (random cell topologies + edges pruning)|
|`mo_prune_201.py`          |NAS-Bench-201          |TF-MOPNAS (edges pruning)|

For example, we want to search on NAS-Bench-201 benchmark:
```shell
$ python mo_prune_201.py
```
**MOENAS + TF-MOENAS**

We will upload the source code in later. However, you can download the results of MOENAS and TF-MOENAS by following the section 1.2.
#### 1.2. Download available results
Download `results.zip` in [here](https://drive.google.com/drive/folders/17p5q3kxK6x3H7Tdf7RLDTg39zIPAxBU5?usp=sharing), unzip and put all folders into ```results``` folder.

### 2. Evaluate
#### 2.1. To generate figures of IGD trends
```shell
$ python <running_file>
```
To conduct evaluate on a specific search space, we just run the corresponding file.
|running_file                        |Benchmark              |         
|:-----------------------------------|:----------------------|
|`visualize_results_101.py`          |NAS-Bench-101          |
|`visualize_results_201.py`          |NAS-Bench-201          |

For example, we want to evaluate the performance (i.e., the IGD indicator) of algorithms on NAS-Bench-201 benchmark:
```shell
$ python visualize_results_201.py
```
#### 2.2. To get the final IGD and the accuracy of the best architecture found
```shell
$ python get_final_performance.py
```

***Note:*** To experiment without changing too much in the source code, we recommend that the *benchmarks' databases* and *experimental results* should be stored in  default folders (i.e., `benchmark_data` and `results`) 

## Results in paper
### NAS-Bench-101
<p align="center">
  <img src="https://user-images.githubusercontent.com/49996342/172910488-c4bf90ed-7ac6-4cbb-be1f-4feb773e3521.jpg" width="350" title="hover text">
  <img src="https://user-images.githubusercontent.com/49996342/172911169-29fdd07b-23f8-435e-86b9-eef7d2b743bc.jpg" width="350" title="hover text">
  <img src="https://user-images.githubusercontent.com/49996342/172912432-0ceddbcf-34cb-4736-8348-1d801f23cb96.JPG" width="600" title="hover text">
  <img src="https://user-images.githubusercontent.com/49996342/172912075-20f38c8f-c5a8-4e23-8876-3dbbbcf3fb7e.JPG" width="600" title="hover text">
</p>

### NAS-Bench-1shot1
<p align="center">
  <img src="https://user-images.githubusercontent.com/49996342/172912663-481c0184-b21b-45bf-9299-996413a8ef56.JPG" width="600" title="hover text">
</p>

### NAS-Bench-201
<p align="center">
  <img src="https://user-images.githubusercontent.com/49996342/172911703-07121a38-56ca-4b26-a562-470ac7e31ff8.jpg" width="350" title="hover text">
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/49996342/172912849-2b004e23-4ea9-4f25-9707-811410d3499b.JPG" width="600" title="hover text">
  <img src="https://user-images.githubusercontent.com/49996342/172912896-2b98c28d-b5c7-4bd4-9221-494596386e9e.JPG" width="600" title="hover text">
</p>

## Acknowledgement
Our source code is inspired by:
- [pymoo: Multi-objective Optimization in Python](https://github.com/anyoptimization/pymoo)
- [NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm](https://github.com/ianwhale/nsga-net)
- [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://github.com/google-research/nasbench)
- [NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search](https://github.com/D-X-Y/NAS-Bench-201)
- [NAS-Bench-1shot1: Benchmarking and Dissecting One-shot Neural Architecture Search](https://github.com/automl/nasbench-1shot1)
- [How Powerful are Performance Predictors in Neural Architecture Search?](https://github.com/automl/NASLib)
