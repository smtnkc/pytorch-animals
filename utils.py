import os
import json
import random
import argparse
import datetime as dt
from str2bool import str2bool
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report


def load_args():
    parser = argparse.ArgumentParser(description='PyTorch Animals')
    parser.add_argument('--seed', default=2020, type=int)
    parser.add_argument('--batch_size', default=14, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--optimizer', default='sgdm', type=str, help='sgdm or adam')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--debug', default=False, type=str2bool)
    parser.add_argument('--pretrained', default=True, type=str2bool)
    parser.add_argument('--device', default='cpu', type=str, help='cpu or cuda')
    parser.add_argument('--t_start', default=None, type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    t_start = dt.datetime.now().replace(microsecond=0)
    args.t_start = t_start.strftime("%Y%m%d_%H%M%S")

    # Reset device to cpu if cuda is not available
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'

    return args

#
#
#
#
#
#

def create_config_file(args):

    config_params = {}  # to dump the run configurations that will be used in plotting
    for param in vars(args):
        config_params[param] = getattr(args, param)

    # Dump run configs to JSON file
    config_path = 'logs/{}_{}.json'.format('pt' if args.pretrained else 'fs', args.t_start)
    with open(config_path, "w") as json_data_file:
        json.dump(config_params, json_data_file)
    fprint('\nCreated config file\t-> {}'.format(config_path), args)

    return config_path



# Plot single image
def display_single(img_folders, normalize, category_names, img_id=-1):
    folder = img_folders['test']
    if img_id == -1:
        img_id = random.randint(0, len(folder))
    tensor = folder.__getitem__(img_id)
    img = tensor[0].permute(1, 2, 0)
    label = tensor[1]
    img_denorm = img * np.array(normalize.std) + np.array(normalize.mean) # denormalization
    plt.figure(figsize=(3, 3))
    plt.imshow(np.clip(img_denorm, 0, 1))
    plt.title(category_names[label])
    plt.show()

#
#
#
#
#
#


# Plot N images
def display_multiple(args, normalize, category_names, data_loaders, N):

    # Get a batch of test data
    batch = next(iter(data_loaders['test']))

    if N > args.batch_size:
        print("N must be less than batch_size={}".format(args.batch_size))
        return

    images, categories = batch
    title = [category_names[x] for x in categories[0:N]]

    # Get a grid of N images from batch
    grid = torchvision.utils.make_grid(images[0:N])

    grid = grid.permute(1, 2, 0)  # needed since PyTorch Tensors are channel-first
    grid = grid * np.array(normalize.std) + np.array(normalize.mean) # denormalization
    grid = np.clip(grid, 0, 1)
    plt.figure(figsize = (N*1.5, 1.5))  # set an arbitrary size
    plt.imshow(grid)
    plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

#
#
#
#
#
#

def get_config(config_path):
    if config_path == '' or not os.path.exists(config_path):
        print('ERROR: set a valid --config_path')
        return

    with open(config_path) as json_data_file:
        configs = json.load(json_data_file)
    return configs

#
#
#
#
#
#


def fprint(string, args, on_console=True):
    if on_console:
        print(string)  # also print on console

    logs_dir = os.path.join(os.path.realpath(''), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logs_path = os.path.join(logs_dir, '{}_{}.txt'.format(
        'pt' if args.pretrained else 'fs', args.t_start))

    with open(logs_path, "a") as f:
        f.write(string + "\n")

#
#
#
#
#
#

def calculate_metrics(preds, labels):
    preds = preds.numpy()
    labels = labels.numpy()

    # fprint(classification_report(labels, preds), args)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    # recall = recall_score(labels, preds, average='weighted')
    # prec = precision_score(labels, preds, average='weighted')

    return acc, f1
