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

    return [ga_results, sa_results, rhc_results, mim_results]


def process_fitness(result_list):
    for result in result_list:
        groups = result.groupby(['param1','param2','param3'])

        fig, ax = plt.subplots()
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
        for name, group in groups:
            print(name)
            print(group)
            ax.plot(group.iterations, group.fitness, marker='o', linestyle='', ms=12, label=name)
        ax.legend()

        plt.show()

        '''
        print(result.head())
        result.plot(x='iterations', y='fitness', kind='line', label=True)
        plt.show()
        '''


def process_positions():

    sorted_results = load_results_dicts(os.path.join(os.curdir, 'results/Four_Peaks'))

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


def process_images():

    sorted_results = load_results_dicts(os.path.join(os.curdir, 'results/Image'))


    problem = OptimizationImage(os.path.join(os.curdir,'img/facebook_logo.png'))

    fig = plt.figure()
    gs = gridspec.GridSpec(4, 4)


    for algo_ind, algo in enumerate(sorted_results.iteritems()):
        name = algo[0]
        print(name)
        outputs = algo[1]

        ax = fig.add_subplot(gs[3, algo_ind])
        ax1 = fig.add_subplot(gs[2, algo_ind])
        ax2 = fig.add_subplot(gs[1, algo_ind])
        ax3 = fig.add_subplot(gs[0, algo_ind])

        max_scores = []
        final_scores = []
        # converge_iters = []
        # full_scores = []
        full_members = []
        for output in outputs:

            members = [res[0] for res in output]
            scores = np.asarray([res[1] for res in output])

            # full_scores.append(scores)
            full_members.append(members)
            final_scores.append(scores[-1])

            max_scores.append((np.max(scores), np.argmax(scores)))

        max_scores = np.asarray(max_scores)
        # print(max_scores.shape)
        best_run_ind = np.argmax(max_scores[:,0], axis=0)

        # print(best_run)
        best_image_ind = int(max_scores[best_run_ind, 1])

        final_plot = full_members[best_run_ind][best_image_ind]
        plot_60 = full_members[best_run_ind][int(best_image_ind*.6)]
        plot_30 = full_members[best_run_ind][int(best_image_ind*.3)]
        plot_0 = full_members[best_run_ind][0]


        ax.imshow(problem.convert_for_plotting(final_plot), cmap='binary')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax1.imshow(problem.convert_for_plotting(plot_60), cmap='binary')
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)

        ax2.imshow(problem.convert_for_plotting(plot_30), cmap='binary')
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)

        ax3.imshow(problem.convert_for_plotting(plot_0), cmap='binary')
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax3.set_title(name.upper())

        if algo_ind == 0:
            ax.set_ylabel('100%')

        # print(pos_tuples[0].shape)
        # ax.scatter(pos_tuples[-1][:, 0], pos_tuples[-1][:, 1], c=colorList[algo_ind*3], label=name.upper())

        # avg_scores = np.mean(np.asarray(pos_tuples), axis=0)
        # print(avg_scores.shape)



    # ax.legend()
    plt.show()





if __name__ == "__main__":

    data_path = os.path.join(os.curdir,'CONTPEAKS')
    ans = load_results_dicts(data_path)
    print('now stage 2')
    process_fitness(ans)
    #process_positions()
    #process_images()