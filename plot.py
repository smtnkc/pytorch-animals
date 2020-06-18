import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch Animals Plot')
parser.add_argument('--csv_path', default='./logs/stats_pretrained_XXX.csv', type=str)
parser.add_argument('--metric', default='acc', type=str)
args = parser.parse_args()


def plot_curves(df, metric):
    v_train = df['train_' + metric]
    v_val = df['val_' + metric]

    plt.title("{} vs {}".format('train', 'val'))
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.plot(range(1, len(df)+1), v_train, label='train')
    plt.plot(range(1, len(df)+1), v_val, label='val')
    plt.ylim((0, 1.0))
    plt.xticks(np.arange(1, len(df)+1, 1.0))
    plt.legend()
    plt.show()

df = pd.read_csv(args.csv_path)
plot_curves(df, metric=args.metric)
