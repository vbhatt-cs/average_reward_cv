import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

color_sequence = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']


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


def plot_1():
    """
    Plot for one step vs multi step
    """
    exp = 1554521162
    path = 'Experiments/lambda_on_policy_control/{}/results.csv'.format(exp)
    res_df = pd.read_csv(path)
    res_df = res_df[res_df['init_rbar'] == 1]

    fig, ax = init_fig()
    text_offsets = [0, -0.005, -0.01, -0.1]

    unique_ls = res_df['lambda'].unique()
    for i, l in enumerate(unique_ls):
        df = res_df[res_df['lambda'] == l]
        xs, ys = get_xy(df, 'alpha', 'reward')

        plt.plot(xs, ys['reward'], color=color_sequence[i])
        plt.fill_between(xs, ys['reward'] - ys['sem_reward'], ys['reward'] + ys['sem_reward'], alpha=0.5,
                         color=color_sequence[i])
        plt.text(xs[-1], ys['reward'][-1:] + text_offsets[i], r'$\lambda$ = {}'.format(l), color=color_sequence[i])

    plt.ylabel('REWARD')
    plt.xlabel(r'$\beta$')
    # plt.legend()
    plt.xscale('log', basex=2)
    # plt.ylim(0.2, 0.4)
    plt.ylim(0, 500)
    plt.show()


# exp_dir = 'Experiments/'
# exp_names1 = ['n_step_on_policy_prediction', 'n_step_off_policy_prediction', 'n_step_cv_prediction']
# exp_names2 = ['lambda_on_policy_prediction', 'lambda_off_policy_prediction', 'lambda_cv_prediction']
# # exp_names1 = ['n_step_on_policy_control', 'n_step_off_policy_control', 'n_step_cv_control']
# # exp_names2 = ['lambda_on_policy_control', 'lambda_off_policy_control', 'lambda_cv_control']
#
# dir_path = exp_dir + exp_names2[1] + '/'
# exp = os.listdir(dir_path)[-2]
# exp_path = dir_path + exp + '/'
#
# res_df = pd.read_csv(exp_path + 'results.csv')
#
# df1 = res_df[res_df['lambda'] == 0]
# df2 = res_df[res_df['lambda'] == 0.5]
# df4 = res_df[res_df['lambda'] == 0.75]
#
# xs = res_df['alpha'].unique()
# y1 = []
# y2 = []
# y4 = []
#
# for x in xs:
#     idx = df1[df1['alpha'] == x]['rmse'].idxmin()
#     y1.append(df1.loc[idx])
#     idx = df2[df2['alpha'] == x]['rmse'].idxmin()
#     y2.append(df2.loc[idx])
#     idx = df4[df4['alpha'] == x]['rmse'].idxmin()
#     y4.append(df4.loc[idx])
#
# y1 = pd.DataFrame(y1)
# y2 = pd.DataFrame(y2)
# y4 = pd.DataFrame(y4)
#
# plt.plot(xs, y1['rmse'])
# plt.fill_between(xs, y1['rmse'] - y1['sem_rmse'], y1['rmse'] + y1['sem_rmse'])
# # plt.text(xs[0], y1['rmse'][:1], r'$\lambda$ = 0')
# plt.text(xs[-1], y1['rmse'][-1:], r'$\lambda$ = 0')
# # plt.text(2 ** (-4.8), 200, r'$\lambda$ = 0')
# plt.plot(xs, y2['rmse'])
# plt.fill_between(xs, y2['rmse'] - y2['sem_rmse'], y2['rmse'] + y2['sem_rmse'])
# # plt.text(xs[0], y2['rmse'][:1], r'$\lambda$ = 0.5')
# plt.text(xs[-1], y2['rmse'][-1:], r'$\lambda$ = 0.5')
# # plt.text(2 ** (-5.9), 470, r'$\lambda$ = 0.5')
# plt.plot(xs, y4['rmse'])
# plt.fill_between(xs, y4['rmse'] - y4['sem_rmse'], y4['rmse'] + y4['sem_rmse'])
# # plt.text(xs[0], y4['rmse'][:1], r'$\lambda$ = 0.75')
# plt.text(xs[-1], y4['rmse'][-1:] - 0.02, r'$\lambda$ = 0.75')
# # plt.text(2 ** (-7), 600, r'n = 4')
#
# # dir_path = exp_dir + exp_names2[2] + '/'
# # exp = os.listdir(dir_path)[-4]
# # exp_path = dir_path + exp + '/'
# #
# # res_df = pd.read_csv(exp_path + 'results.csv')
# #
# # df1 = res_df[res_df['lambda'] == 0]
# # df2 = res_df[res_df['lambda'] == 0.5]
# # df4 = res_df[res_df['lambda'] == 0.75]
# #
# # xs = res_df['alpha'].unique()
# # y1 = []
# # y2 = []
# # y4 = []
# #
# # for x in xs:
# #     idx = df1[df1['alpha'] == x]['rmse'].idxmin()
# #     y1.append(df1.loc[idx])
# #     idx = df2[df2['alpha'] == x]['rmse'].idxmin()
# #     y2.append(df2.loc[idx])
# #     idx = df4[df4['alpha'] == x]['rmse'].idxmin()
# #     y4.append(df4.loc[idx])
# #
# # y1 = pd.DataFrame(y1)
# # y2 = pd.DataFrame(y2)
# # y4 = pd.DataFrame(y4)
# #
# # plt.plot(xs, y1['rmse'])
# # plt.fill_between(xs, y1['rmse'] - y1['sem_rmse'], y1['rmse'] + y1['sem_rmse'])
# # plt.text(xs[-1], y1['rmse'][-1:], r'$\lambda$ = 0')
# # # plt.text(2 ** (-7), 200, r'$\lambda$ = 0')
# # # plt.text(xs[0], y1['rmse'][:1], r'cv')
# # plt.plot(xs, y2['rmse'])
# # plt.fill_between(xs, y2['rmse'] - y2['sem_rmse'], y2['rmse'] + y2['sem_rmse'])
# # plt.text(xs[-1], y2['rmse'][-1:], r'$\lambda$ = 0.5')
# # # plt.text(2 ** (-7.2), 250, r'$\lambda$ = 0.5')
# # # plt.text(xs[0], y2['rmse'][:1], r'cv')
# # # plt.text(2 ** (-4), 0.29, r'cv')
# # plt.plot(xs, y4['rmse'])
# # plt.fill_between(xs, y4['rmse'] - y4['sem_rmse'], y4['rmse'] + y4['sem_rmse'])
# # plt.text(xs[-1], y4['rmse'][-1:] - 0.02, r'$\lambda$ = 0.75')
# # # plt.text(2 ** (-5.8), 280, r'$\lambda$ = 0.75')
# # # plt.text(xs[0], y4['rmse'][:1], r'cv')
# # # plt.text(2 ** (-4.1), 0.28, r'cv')
# #
# # exp = os.listdir(dir_path)[-1]
# # exp_path = dir_path + exp + '/'
# #
# # res_df = pd.read_csv(exp_path + 'results.csv')
# #
# # df1 = res_df[res_df['lambda'] == 0]
# # df2 = res_df[res_df['lambda'] == 0.5]
# # df4 = res_df[res_df['lambda'] == 0.75]
# #
# # xs = res_df['alpha'].unique()
# # y1 = []
# # y2 = []
# # y4 = []
# #
# # for x in xs:
# #     idx = df1[df1['alpha'] == x]['rmse'].idxmin()
# #     y1.append(df1.loc[idx])
# #     idx = df2[df2['alpha'] == x]['rmse'].idxmin()
# #     y2.append(df2.loc[idx])
# #     idx = df4[df4['alpha'] == x]['rmse'].idxmin()
# #     y4.append(df4.loc[idx])
# #
# # y1 = pd.DataFrame(y1)
# # y2 = pd.DataFrame(y2)
# # y4 = pd.DataFrame(y4)
# #
# # plt.plot(xs, y1['rmse'])
# # plt.fill_between(xs, y1['rmse'] - y1['sem_rmse'], y1['rmse'] + y1['sem_rmse'])
# # plt.text(xs[0], y1['rmse'][:1], r'cv_rbar')
# # # plt.text(xs[0], y1['rmse'][:1], r'n = 1')
# # plt.plot(xs, y2['rmse'])
# # plt.fill_between(xs, y2['rmse'] - y2['sem_rmse'], y2['rmse'] + y2['sem_rmse'])
# # plt.text(xs[0], y2['rmse'][:1], r'cv_rbar')
# # # plt.text(xs[-1], y2['rmse'][-1:], r'$\lambda$ = 0.5')
# # plt.plot(xs, y4['rmse'])
# # plt.fill_between(xs, y4['rmse'] - y4['sem_rmse'], y4['rmse'] + y4['sem_rmse'])
# # # plt.text(xs[0], y4['rmse'][:1], r'cv_rbar')
# # plt.text(2 ** (-4), 0.28, r'cv_rbar')
# # # plt.text(xs[-1], y4['rmse'][-1:], r'$\lambda$ = 0.75')
#
# plt.title(r'RMSE vs $\alpha$')
# plt.ylabel('RMSE')
# plt.xlabel(r'$\alpha$')
# # plt.legend()
# plt.xscale('log', basex=2)
# plt.ylim(0, 0.3)
# # plt.ylim(125, 325)
# plt.show()


# exp = 1554771113
# path = 'Experiments/lambda_on_policy_prediction/{}/results.csv'.format(exp)
# res_df = pd.read_csv(path)
# res_df = res_df[res_df['init_rbar'] == 0.01]
# df_sorted = res_df.sort_values('rmse')
#
# df1 = res_df[res_df['lambda'] == 0]
# df2 = res_df[res_df['lambda'] == 0.5]
# df4 = res_df[res_df['lambda'] == 0.75]
#
# xs = res_df['beta'].unique()
# y1 = []
# y2 = []
# y4 = []
#
# for x in xs:
#     idx = df1[df1['beta'] == x]['rmse'].idxmin()
#     y1.append(df1.loc[idx])
#     idx = df2[df2['beta'] == x]['rmse'].idxmin()
#     y2.append(df2.loc[idx])
#     idx = df4[df4['beta'] == x]['rmse'].idxmin()
#     y4.append(df4.loc[idx])
#
# y1 = pd.DataFrame(y1)
# y2 = pd.DataFrame(y2)
# y4 = pd.DataFrame(y4)
#
# plt.plot(xs, y1['rmse'])
# plt.fill_between(xs, y1['rmse'] - y1['sem_rmse'], y1['rmse'] + y1['sem_rmse'])
# # plt.text(xs[0], y1['rmse'][:1], r'$\lambda$ = 0')
# plt.text(xs[-1], y1['rmse'][-1:], r'$\lambda$ = 0')
# # plt.text(2 ** (-4.8), 200, r'$\lambda$ = 0')
# plt.plot(xs, y2['rmse'])
# plt.fill_between(xs, y2['rmse'] - y2['sem_rmse'], y2['rmse'] + y2['sem_rmse'])
# # plt.text(xs[0], y2['rmse'][:1], r'$\lambda$ = 0.5')
# plt.text(xs[-1], y2['rmse'][-1:], r'$\lambda$ = 0.5')
# # plt.text(2 ** (-5.9), 470, r'$\lambda$ = 0.5')
# plt.plot(xs, y4['rmse'])
# plt.fill_between(xs, y4['rmse'] - y4['sem_rmse'], y4['rmse'] + y4['sem_rmse'])
# # plt.text(xs[0], y4['rmse'][:1], r'$\lambda$ = 0.75')
# plt.text(xs[-1], y4['rmse'][-1:], r'$\lambda$ = 0.75')
# # plt.text(2 ** (-7), 600, r'n = 4')
#
# plt.title(r'RMSE vs $\beta$')
# plt.ylabel('RMSE')
# plt.xlabel(r'$\beta$')
# # plt.legend()
# plt.xscale('log', basex=2)
# plt.ylim(0.2, 0.4)
# # plt.ylim(125, 325)
# plt.show()

if __name__ == '__main__':
    plot_1()
