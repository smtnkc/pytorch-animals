""" This file includes global constant variables """

import os
from torchvision import transforms

DATA_DIR = os.path.join(os.path.realpath(''), 'data')
LOG_DIR = os.path.join(os.path.realpath(''), 'logs')
PLOT_DIR = os.path.join(os.path.realpath(''), 'plots')
MODEL_DIR = os.path.join(os.path.realpath(''), 'models')
CATEGORIES = ["bear", "elephant", "leopard", "zebra"]
NUM_CATEGORIES = len(CATEGORIES)

# All pre-trained models expect input images normalized in the same way
# For details: https://pytorch.org/docs/master/torchvision/models.html
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
PHASES = ['train', 'val', 'test']
