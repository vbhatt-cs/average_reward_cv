import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

color_sequence = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
cur_color = 0
params = {'font.size': 13, 'axes.labelsize': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13}
mpl.rcParams.update(params)


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
    exp = 1554705207
    path = 'Experiments/lambda_on_policy_prediction/{}/results.csv'.format(exp)
    res_df = pd.read_csv(path)
    # res_df = res_df[res_df['init_rbar'] == 0.1]

    fig, ax = init_fig()
    text_x = [2 ** (-0), 2 ** (-0), 2 ** (-0), 2 ** (-1)]
    text_y = [0.3, 0.23, 0.335, 0.45]

    plot_ls(res_df, text_x, text_y, 'alpha', 'rmse')

    plt.ylabel('RMSE after 1000 steps')
    # plt.legend()
    plt.xscale('log', basex=2)
    plt.ylim(0.1, 0.5)
    # plt.ylim(0, 500)
    plt.show()
    cur_color = 0


def plot_2():
    """
    Plot graphs from 2 experiments.
    """
    global cur_color
    exp1 = 1555378416
    path1 = 'Experiments/lambda_cv_prediction/{}/results.csv'.format(exp1)
    exp2 = 1555378414
    path2 = 'Experiments/lambda_cv_prediction/{}/results.csv'.format(exp2)
    fig, ax = init_fig()

    res_df = pd.read_csv(path1)
    res_df = res_df[res_df['lambda'].isin([0, 0.5])]
    text_x = [2 ** (-1.5), 2 ** (-1), 2 ** (-4.2), 2 ** (-5.5)]
    text_y = [0.16, 0.28, 0.33, 0.34]
    plot_ls(res_df, text_x, text_y, 'alpha', 'rmse')

    res_df = pd.read_csv(path2)
    res_df = res_df[res_df['lambda'].isin([0, 0.5])]
    text_x = [2 ** (-1), 2 ** (-1), 2 ** (-4.2), 2 ** (-5.5)]
    text_y = [0.2, 0.31, 0.28, 0.29]
    plot_ls(res_df, text_x, text_y, 'alpha', 'rmse')

    plt.text(2 ** (-1.5), 0.145, r'(cv $\bar{R}$)', color=color_sequence[0])
    plt.text(2 ** (-1), 0.265, r'(cv $\bar{R}$)', color=color_sequence[1])
    plt.text(2 ** (-1), 0.19, r'(cv)', color=color_sequence[2])
    plt.text(2 ** (-1), 0.3, r'(cv)', color=color_sequence[3])

    plt.ylabel('RMSE after 1000 steps')
    # plt.legend()
    plt.xscale('log', basex=2)
    plt.ylim(0.12, 0.36)
    # plt.ylim(0, 500)
    plt.show()
    cur_color = 0


if __name__ == '__main__':
    # plot_1()
    plot_2()
