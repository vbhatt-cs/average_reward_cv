import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

color_sequence = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
cur_color = 0


def init_fig():
    fig, ax = plt.subplots(1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax


def get_xy(df, col, metric):
    xs = df[col].unique()
    ys = []

    for x in xs:
        idx = df[df[col] == x][metric].idxmin()
        ys.append(df.loc[idx])

    ys = pd.DataFrame(ys)
    return xs, ys


def plot_ls(res_df, text_x, text_y, col, metric):
    global cur_color
    unique_ls = res_df['lambda'].unique()
    for i, l in enumerate(unique_ls):
        df = res_df[res_df['lambda'] == l]
        xs, ys = get_xy(df, col, metric)

        plt.plot(xs, ys[metric], color=color_sequence[cur_color])
        plt.fill_between(xs, ys[metric] - ys['sem_' + metric], ys[metric] + ys['sem_' + metric], alpha=0.5,
                         color=color_sequence[cur_color])
        plt.text(text_x[i], text_y[i], r'$\lambda$ = {}'.format(l), color=color_sequence[cur_color])
        cur_color += 1

    plt.xlabel(r'$\{}$'.format(col))


def plot_1():
    """
    Plot graph from 1 experiment.
    """
    global cur_color
    exp = 1554774262
    path = 'Experiments/lambda_on_policy_prediction/{}/results.csv'.format(exp)
    res_df = pd.read_csv(path)
    res_df = res_df[res_df['init_rbar'] == 1]

    fig, ax = init_fig()
    text_x = [2 ** (-1), 2 ** (-1), 2 ** (-4.2), 2 ** (-5.5)]
    text_y = [0.3, 0.345, 0.33, 0.34]

    plot_ls(res_df, text_x, text_y, 'alpha', 'rmse')

    plt.ylabel('RMSE after 1000 steps')
    # plt.legend()
    plt.xscale('log', basex=2)
    plt.ylim(0.27, 0.35)
    # plt.ylim(0, 500)
    plt.show()
    cur_color = 0


def plot_2():
    """
    Plot graphs from 2 experiments.
    """
    exp1 = 1555222309
    path1 = 'Experiments/lambda_cv_prediction/{}/results.csv'.format(exp1)
    exp2 = 1555221399
    path2 = 'Experiments/lambda_cv_prediction/{}/results.csv'.format(exp2)
    fig, ax = init_fig()

    res_df = pd.read_csv(path1)
    text_x = [2 ** (-3.8), 2 ** (-5.5), 2 ** (-4.2), 2 ** (-5.5)]
    text_y = [0.27, 0.268, 0.33, 0.34]
    plot_ls(res_df, text_x, text_y, 'alpha', 'rmse')

    res_df = pd.read_csv(path2)
    text_x = [2 ** (-6), 2 ** (-7.5), 2 ** (-4.2), 2 ** (-5.5)]
    text_y = [0.32, 0.34, 0.28, 0.29]
    plot_ls(res_df, text_x, text_y, 'alpha', 'rmse')

    plt.text(2 ** (-6), 0.313, r'(full $\bar{R}$)', color=color_sequence[2])
    plt.text(2 ** (-7.5), 0.333, r'(full $\bar{R}$)', color=color_sequence[3])

    plt.ylabel('RMSE after 1000 steps')
    plt.legend()
    plt.xscale('log', basex=2)
    plt.ylim(0.15, 0.5)
    # plt.ylim(0, 500)
    plt.show()


if __name__ == '__main__':
    # plot_1()
    plot_2()
