import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import cfg
from utils import load_json_args

def generate_plots(json_path):

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

        plt.title("{} vs {} (opt={}, lr={})".format(
            'train', 'val', json_args['optimizer'], json_args['lr']))
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.plot(range(1, len(stats_df)+1), v_train, label='train')
        plt.plot(range(1, len(stats_df)+1), v_val, label='val')
        # plt.ylim((0, 1.0))
        plt.xticks(np.arange(1, len(stats_df)+1, 1.0))
        plt.legend()

        # create plot png
        if not os.path.exists(sub_dump_dir):
            os.makedirs(sub_dump_dir)
        plot_name = 'plot_{}.png'.format(metric)
        plot_path = os.path.join(sub_dump_dir, plot_name)
        plt.savefig(plot_path)
        plt.close()
        print('Saved {} plot\t-> {}'.format(metric, plot_path))


def main():
    parser = argparse.ArgumentParser(description='PyTorch Animals Plot')
    parser.add_argument('--json_path', default='', type=str)
    args = parser.parse_args()
    generate_plots(args.json_path)


if __name__ == "__main__":
    main()
