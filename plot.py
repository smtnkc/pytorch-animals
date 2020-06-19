import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_plots(JSON_PATH):

    if JSON_PATH == '' or not os.path.exists(JSON_PATH):
        print('ERROR: set a valid --json_path')
        return

    with open(JSON_PATH) as json_data_file:
        json_data = json.load(json_data_file)

    csv_name = '{}_{}.csv'.format('pt' if json_data['pretrained'] else 'fs', json_data['t_start'])
    csv_path = os.path.join(os.path.realpath(''), 'logs', csv_name)

    if not os.path.exists(csv_path):
        print('ERROR: Cannot found {}'.format(csv_path))
        return

    stats_df = pd.read_csv(csv_path)

    for metric in ['acc', 'loss']:
        v_train = stats_df['train_' + metric]
        v_val = stats_df['val_' + metric]

        plt.title("{} vs {} (opt={}, lr={})".format(
            'train', 'val', json_data['optimizer'], json_data['lr']))
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.plot(range(1, len(stats_df)+1), v_train, label='train')
        plt.plot(range(1, len(stats_df)+1), v_val, label='val')
        # plt.ylim((0, 1.0))
        plt.xticks(np.arange(1, len(stats_df)+1, 1.0))
        plt.legend()

        plots_dir = os.path.join(os.path.realpath(''), 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        plot_name = '{}_{}_{}.png'.format(
            'pt' if json_data['pretrained'] else 'fs', json_data['t_start'], metric)
        plot_path = os.path.join(plots_dir, plot_name)
        plt.savefig(plot_path)
        plt.close()
        print('Saved {} plot as {}'.format(metric, plot_path))


def main():
    parser = argparse.ArgumentParser(description='PyTorch Animals Plot')
    parser.add_argument('--json_path', default='', type=str)
    args = parser.parse_args()
    generate_plots(args.json_path)


if __name__ == "__main__":
    main()
