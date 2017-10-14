__author__ = 'Jeff'
import os
import pickle as pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


colorList = [
            [255./255., 51./255., 51./255.],
            [255./255., 153./255., 51./255.],
            [255./255., 255./255, 51./255],
            [153./255., 255./255, 51./255],
            [51./255., 255./255, 51./255],
            [51./255., 255./255, 153./255],
            [51./255., 255./255, 255./255],
            [51./255., 153./255, 255./255],
            [51./255., 51./255, 255./255],
            [153./255., 51./255, 255./255],
            [255./255., 51./255, 255./255],
            [255./255., 51./255, 153./255],
            [160./255., 160./255, 160./255]
            ]


def load_results_dicts(results_path):

    files = os.listdir(results_path)

    results_dicts = []
    for file_name in files:

        full_path = os.path.join(results_path, file_name)

        with open(full_path, 'rb') as file_obj:
            results_dicts.append(pd.read_csv(file_obj))
    a = pd.concat(results_dicts)
    results_group = a.groupby('algo', as_index=False)['algo','trial','iterations','param1','param2','param3','fitness','time','fevals']

    ga_results = results_group.get_group('GA').groupby(['iterations','param1','param2','param3'],as_index=False)['algo','fitness','time','fevals'].mean()
    sa_results = results_group.get_group('SA').groupby(['iterations','param1','param2','param3'],as_index=False)['algo','fitness','time','fevals'].mean()
    rhc_results = results_group.get_group('RHC').groupby(['iterations','param1','param2','param3'],as_index=False)['algo','fitness','time','fevals'].mean()
    mim_results = results_group.get_group('MIMIC').groupby(['iterations','param1','param2','param3'],as_index=False)['algo','fitness','time','fevals'].mean()

    return [('GA',ga_results), ('SA',sa_results), ('RHC',rhc_results), ('MIMIC',mim_results)]


def process_fitness(result_list):
    for result in result_list:
        groups = result[1].groupby(['param1','param2','param3'])

        fig, ax = plt.subplots()
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
        for name, group in groups:
            print(name)
            print(group)
            ax.plot(group.iterations, group.fitness, marker='o', linestyle='', ms=12, label=name)
            ax.legend()
        ax.set_title('Fitness Function ' + result[0])

    plt.show()


def process_time(result_list):
    for result in result_list:
        groups = result[1].groupby(['param1','param2','param3'])

        fig, ax = plt.subplots()
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
        for name, group in groups:
            print(name)
            print(group)
            ax.plot(group.iterations, group.time, marker='o', linestyle='', ms=12, label=name)
            ax.legend()
        ax.set_title('Time Complexity ' + result[0])

    plt.show()



def process_positions():

    sorted_results = load_results_dicts(os.path.join(os.curdir, 'TSP'))

    print(sorted_results.keys())

    problem = OptimizationSurface(four_peaks, (0, 256), (0, 256))

    fig = plt.figure()


    for algo_ind, algo in enumerate(sorted_results.iteritems()):
        name = algo[0]
        print(name)
        outputs = algo[1]

        ax = fig.add_subplot(1, 4, algo_ind+1)
        ax.contour(problem.x, problem.y, problem.z, cmap='jet', zorder=0)

        # max_scores = []
        # final_scores = []
        # converge_iters = []
        # full_scores = []
        pos_tuples = []
        for output in outputs:

            members = [res[0] for res in output]

            # print(members)

            pos_tuples.append(np.asarray([[bin_list_to_int(member[:len(member)/2]),
                                           bin_list_to_int(member[len(member)/2:]),
                                           problem.evaluate(member)[0][0]] for member in members]))

            if len(pos_tuples) == 1:
                ax.scatter(pos_tuples[-1][:, 0], pos_tuples[-1][:, 1], c=colorList[algo_ind*3], s=50, alpha=0.1,
                           zorder=1)

            else:
                ax.scatter(pos_tuples[-1][:, 0], pos_tuples[-1][:, 1], c=colorList[algo_ind*3], s=50, alpha=0.1,
                           zorder=1)

            ax.set_title(name.upper())

        # print(pos_tuples[0].shape)
        # ax.scatter(pos_tuples[-1][:, 0], pos_tuples[-1][:, 1], c=colorList[algo_ind*3], label=name.upper())

        # avg_scores = np.mean(np.asarray(pos_tuples), axis=0)
        # print(avg_scores.shape)


    #
    # ax.legend()
    plt.show()


def process_nn_result(results_path):
    files = os.listdir(results_path)
    full_path = os.path.join(results_path, 'BACK_LOG.xls')
    results_dicts = []
    for sheet in ['BACK', 'RHC', 'SA', 'GA']:
        a = pd.read_excel(full_path, sheetname=sheet)
        results_dicts.append(a)
    a = pd.concat(results_dicts)
    results_group = a.groupby('algo', as_index=False)['algo','iteration', 'MSE_trg', 'MSE_tst', 'acc_trg', 'acc_tst', 'elapsed', 'param1', 'param2', 'param3']
    #print(results_group)

    backprop_results = results_group.get_group('BACK').groupby(['iteration','param1','param2','param3'],as_index=False)['algo','MSE_trg', 'MSE_tst', 'acc_trg', 'acc_tst', 'elapsed'].mean()
    rhc_results = results_group.get_group('RHC').groupby(['iteration','param1','param2','param3'], as_index=False)['algo', 'MSE_trg', 'MSE_tst', 'acc_trg', 'acc_tst', 'elapsed'].mean()
    sa_results = results_group.get_group('SA').groupby(['iteration','param1','param2','param3'], as_index=False)['algo', 'MSE_trg', 'MSE_tst', 'acc_trg', 'acc_tst', 'elapsed'].mean()
    ga_results = results_group.get_group('GA').groupby(['iteration','param1','param2','param3'], as_index=False)['algo', 'MSE_trg', 'MSE_tst', 'acc_trg', 'acc_tst', 'elapsed'].mean()
    print(ga_results)
    return [('back',backprop_results), ('rhc',rhc_results), ('sa',sa_results), ('ga',ga_results)]



def process_nn_perform(result_list):
    for result in result_list:

        groups = result[1].groupby(['param1','param2','param3'])

        fig, ax = plt.subplots()
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
        for name, group in groups:
            print(name)
            print(group)
            ax.plot(group.iteration, group.acc_trg, marker='o', linestyle='-', ms=12, label='train set '+str(name))
            ax.plot(group.iteration, group.acc_tst, marker='o', linestyle='-', ms=12, label='val set '+str(name))
        ax.set_title('Fitness Function ' + result[0])
        ax.legend()

    plt.show()


def process_nn_time(result_list):
    for result in result_list:
        groups = result[1].groupby(['param1','param2','param3'])

        fig, ax = plt.subplots()
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
        for name, group in groups:
            print(name)
            print(group)
            ax.plot(group.iteration, group.elapsed, marker='o', linestyle='-', ms=12, label=name)
        ax.set_title('Fitness Function ' + result[0])
        ax.legend()

    plt.show()




if __name__ == "__main__":
    data_path = os.path.join(os.curdir,'FLIPFLOP') ## this line to be changed to do different analysis
    ans = load_results_dicts(data_path)
    #ans = process_nn_result(data_path)
    print('now stage 2')
    process_fitness(ans)
    #process_time(ans)
    #process_nn_perform(ans)
    #process_nn_time(ans)