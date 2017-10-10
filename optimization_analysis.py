__author__ = 'Jeff'
import os
import pickle as pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

#from AnalyzeData import colorList
#from FitnessFunctions import OptimizationSurface, four_peaks, bin_list_to_int, OptimizationImage


def load_results_dicts(results_path):

    files = os.listdir(results_path)

    results_dicts = []
    for file_name in files:

        full_path = os.path.join(results_path, file_name)

        with open(full_path, 'rb') as file_obj:
            results_dicts.append(pd.read_csv(file_obj))
    print(results_dicts)
'''
    ga_results = []
    sa_results = []
    rhc_results = []
    mim_results = []
    for iteration in results_dicts:
        ga_results.append(iteration['ga'])
        sa_results.append(iteration['sa'])
        rhc_results.append(iteration['rhc'])
        mim_results.append(iteration['mim'])

    sorted_results = dict(ga=ga_results, sa=sa_results, rhc=rhc_results, mim=mim_results)

    return sorted_results
'''

def process_fitness(data_path):

    sorted_results = load_results_dicts(data_path)



    fig_final = plt.figure()
    ax_final = fig_final.add_subplot(111)

    fig_max = plt.figure()
    ax_max = fig_max.add_subplot(111)

    fig_curves = plt.figure()
    ax_curves = fig_curves.add_subplot(111)
    for algo_ind, algo in enumerate(sorted_results.iteritems()):
        name = algo[0]
        outputs = algo[1]

        max_scores = []
        final_scores = []
        converge_iters = []
        full_scores = []
        for output in outputs:

            members = [res[0] for res in output]
            scores = np.asarray([res[1] for res in output])


            full_scores.append(scores)
            final_scores.append(scores[-1])
            max_scores.append(np.max(scores))

            previous = 0
            for score_ind, score_val in enumerate(scores[::-1]):
                if previous > score_val:
                    break
                previous = score_val

            print(score_ind)
            converge_iters.append(scores.shape[0]-score_ind)





        ax_final.scatter(converge_iters, final_scores, c=colorList[algo_ind], alpha=0.8, s=50, label=name.upper())
        ax_max.scatter(final_scores, max_scores, c=colorList[algo_ind], alpha=0.8, s=50, label=name.upper())

        avg_scores = np.mean(np.asarray(full_scores), axis=0).flatten()

        ax_curves.plot(avg_scores, c=colorList[algo_ind], alpha=0.9, linewidth=2, label=name.upper())



    ax_final.legend(loc='best').draw_frame(False)

    ax_max.legend(loc='lower right').draw_frame(False)

    ax_curves.legend(loc='best').draw_frame(False)
    plt.show()


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

    # process_fitness(data_path)
    # process_positions()
    #process_images()