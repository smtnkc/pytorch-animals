import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import cfg
from utils import load_json_args

def generate_plots(json_path, fig_width=15):

    json_args = load_json_args(json_path)
    if json_args is None:
        return

    sub_dump_dir = os.path.join(cfg.DUMP_DIR, '{}_{}').format(
        'pt' if json_args['pretrained'] else 'fs', json_args['t_start'])

    stats_path = os.path.join(sub_dump_dir, 'stats.csv')

    if not os.path.exists(stats_path):
        print('ERROR: Cannot found {}'.format(stats_path))
        return

    stats_df = pd.read_csv(stats_path)

    for metric in ['acc', 'loss']:
        v_train = stats_df['train_' + metric]
        v_val = stats_df['val_' + metric]

        plt.figure(figsize=(fig_width, 5))
        plt.title("opt={}, lr={}, wd={}".format(
            json_args['optimizer'], json_args['lr'], json_args['weight_decay']))
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.plot(range(1, len(stats_df)+1), v_train, label='train')
        plt.plot(range(1, len(stats_df)+1), v_val, label='val')
        if metric == 'acc':
            plt.ylim((0, 1.0))
        plt.xticks(np.arange(1, len(stats_df)+1, 1.0))
        plt.legend()

        # create plot png
        sub_plot_dir = os.path.join(cfg.PLOT_DIR, metric)
        if not os.path.exists(sub_plot_dir):
            os.makedirs(sub_plot_dir)
        plot_name = '{}_{}.png'.format('pt' if json_args['pretrained'] else 'fs', json_args['t_start'])
        plot_path = os.path.join(sub_plot_dir, plot_name)
        plt.savefig(plot_path, bbox_inches='tight', transparent="True", pad_inches=0.1)
        plt.close()
        print('Saved {} plot\t-> {}'.format(metric, plot_path))


def generate_plots_compare(json_path, json_path_compare, fig_width=15):

    json_args = load_json_args(json_path)
    json_args_compare = load_json_args(json_path_compare)
    if json_args is None or json_args_compare is None:
        return

    sub_dump_dir = os.path.join(cfg.DUMP_DIR, '{}_{}').format(
        'pt' if json_args['pretrained'] else 'fs', json_args['t_start'])

    sub_dump_dir_compare = os.path.join(cfg.DUMP_DIR, '{}_{}').format(
        'pt' if json_args_compare['pretrained'] else 'fs', json_args_compare['t_start'])

    stats_path = os.path.join(sub_dump_dir, 'stats.csv')
    stats_path_compare = os.path.join(sub_dump_dir_compare, 'stats.csv')

    if not os.path.exists(stats_path):
        print('ERROR: Cannot found {}'.format(stats_path))
        return
    if not os.path.exists(stats_path_compare):
        print('ERROR: Cannot found {}'.format(stats_path_compare))
        return

    stats_df = pd.read_csv(stats_path)
    stats_df_compare = pd.read_csv(stats_path_compare)
    n_epochs = len(stats_df)+1
    n_epochs_compare = len(stats_df_compare)+1
    n_epochs_max = max(n_epochs, n_epochs_compare)

    for metric in ['acc', 'loss']:
        val = stats_df['val_' + metric]
        val_compare = stats_df_compare['val_' + metric]

        plt.figure(figsize=(fig_width, 5))
        plt.title("opt={}, lr={}, wd={}".format(
            json_args['optimizer'], json_args['lr'], json_args['weight_decay']))
        plt.xlabel('epoch')
        plt.ylabel(metric)

        plt.plot(range(1, n_epochs), val, label='pt' if json_args['pretrained'] else 'fs')
        plt.plot(range(1, n_epochs_compare), val_compare, label='pt' if json_args_compare['pretrained'] else 'fs')

        plt.xticks(np.arange(1, n_epochs_max, 1.0))
        if metric == 'acc':
            plt.ylim((0, 1.0))
        plt.legend()

        # create plot png
        sub_plot_dir = os.path.join(cfg.PLOT_DIR, metric)
        if not os.path.exists(sub_plot_dir):
            os.makedirs(sub_plot_dir)
        plot_name = 'compare_{}_{}_{}_{}.png'.format(
            'pt' if json_args['pretrained'] else 'fs', json_args['t_start'],
            'pt' if json_args_compare['pretrained'] else 'fs', json_args_compare['t_start'])
        plot_path = os.path.join(sub_plot_dir, plot_name)
        plt.savefig(plot_path, bbox_inches='tight', transparent="True", pad_inches=0.1)
        plt.close()
        print('Saved {} plot\t-> {}'.format(metric, plot_path))



def main():
    parser = argparse.ArgumentParser(description='PyTorch Animals Plot')
    parser.add_argument('--json_path', default='', type=str)
    parser.add_argument('--json_path_compare', default='', type=str)
    parser.add_argument('--fig_width', default=15, type=int)
    args = parser.parse_args()

    if args.json_path_compare == '':
        generate_plots(args.json_path, args.fig_width)
    else:
        generate_plots_compare(args.json_path, args.json_path_compare, args.fig_width)


if __name__ == "__main__":
    main()
