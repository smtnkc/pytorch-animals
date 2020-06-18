import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report

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

def fprint(string, args, on_console=False):
    if on_console:
        print(string)  # also print on console

    file_name = 'log_{}.txt'.format(args.t_start)
    # example file_name: log_20200609_164520.txt

    log_file = os.path.join('logs', file_name)

    with open(log_file, "a") as f:
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

    # fprint(classification_report(labels, preds), args, True)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    prec = precision_score(labels, preds, average='weighted')

    return acc, f1
