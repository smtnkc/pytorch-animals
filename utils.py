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

import cfg


def load_args():
    parser = argparse.ArgumentParser(description='PyTorch Animals')
    parser.add_argument('--seed', default=cfg.SEED, type=int)
    parser.add_argument('--batch_size', default=14, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--optimizer', default='sgdm', type=str, help='sgdm or adam')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--display_images', default=False, type=str2bool)
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

def init_random_seeds(seed):
    """ For reproducibility of results """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

#
#
#
#
#
#


def get_sub_dump_dir(args):
    sub_dump_dir = os.path.join(cfg.DUMP_DIR, '{}_{}').format(
        'pt' if args.pretrained else 'fs', args.t_start)
    if not os.path.exists(sub_dump_dir):
        os.makedirs(sub_dump_dir)

    return sub_dump_dir

#
#
#
#
#
#


def export_args(args):

    json_args = {}  # to dump the running args that will be used in plotting
    for v in vars(args):
        json_args[v] = getattr(args, v)

    # Write args to json file
    sub_dump_dir = get_sub_dump_dir(args)
    json_path = os.path.join(sub_dump_dir, 'args.json')

    with open(json_path, "w") as json_file:
        json.dump(json_args, json_file)
    fprint('Created json args file\t-> {}\n'.format(json_path), args)

    return json_path



# Plot single image
def display_single(args, img_folders, img_id=-1):
    folder = img_folders['test']
    if img_id == -1:
        random.seed(args.seed)
        img_id = random.randint(0, len(folder))
    tensor = folder.__getitem__(img_id)
    img = tensor[0].permute(1, 2, 0)
    category_id = tensor[1]
    img_denorm = img * np.array(cfg.NORMALIZE.std) + np.array(cfg.NORMALIZE.mean) # denormalization
    plt.figure(figsize=(3, 3))
    plt.imshow(np.clip(img_denorm, 0, 1))
    plt.title(cfg.CATEGORIES[category_id])
    plt.show()

#
#
#
#
#
#


# Plot N images
def display_multiple(args, data_loaders, N):

    # Get a batch of test data
    batch = next(iter(data_loaders['test']))

    if N > args.batch_size:
        print("N must be less than batch_size={}".format(args.batch_size))
        return

    images, category_ids = batch
    title = [cfg.CATEGORIES[x] for x in category_ids[0:N]]

    # Get a grid of N images from batch
    grid = torchvision.utils.make_grid(images[0:N])

    grid = grid.permute(1, 2, 0)  # needed since PyTorch Tensors are channel-first
    grid = grid * np.array(cfg.NORMALIZE.std) + np.array(cfg.NORMALIZE.mean) # denormalization
    grid = np.clip(grid, 0, 1)
    plt.figure(figsize=(N*1.5, 1.5))  # set an arbitrary size
    plt.imshow(grid)
    plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

#
#
#
#
#
#

def load_json_args(json_path):
    if json_path == '' or not os.path.exists(json_path):
        print('ERROR: set a valid --json_path')
        return

    with open(json_path) as json_file:
        json_args = json.load(json_file)
    return json_args

#
#
#
#
#
#


def fprint(string, args, on_console=True):
    if on_console:
        print(string)  # also print on console

    sub_dump_dir = get_sub_dump_dir(args)
    logs_path = os.path.join(sub_dump_dir, 'logs.txt')

    with open(logs_path, "a") as log_file:
        log_file.write(string + "\n")

#
#
#
#
#
#

def calculate_metrics(preds, category_ids):
    preds = preds.numpy()
    category_ids = category_ids.numpy()

    # print(classification_report(category_ids, preds))
    acc = accuracy_score(category_ids, preds)
    f1 = f1_score(category_ids, preds, average='weighted')
    # recall = recall_score(category_ids, preds, average='weighted')
    # prec = precision_score(category_ids, preds, average='weighted')

    return acc, f1
