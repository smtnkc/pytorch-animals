from __future__ import division
from __future__ import print_function

import os
import platform
import argparse
import datetime as dt
from str2bool import str2bool
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder as IF
from torch.utils.data import DataLoader as DL

from utils import display_single, display_multiple, fprint
from model_helper import initialize_model, train_model, test_model, predict

print("Python Version:", platform.python_version())
print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__.split('a')[0])

parser = argparse.ArgumentParser(description='PyTorch Animals')
parser.add_argument('--seed', default=2020, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--input_size', default=224, type=int)
parser.add_argument('--show_plots', default=False, type=str2bool)
parser.add_argument('--use_pretrained', default=True, type=str2bool)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--t_start', default=None, type=str, help=argparse.SUPPRESS)

args = parser.parse_args()

t_start = dt.datetime.now().replace(microsecond=0)
args.t_start = t_start.strftime("%Y%m%d_%H%M%S")

INPUT_SIZE = args.input_size
DATA_DIR = os.path.join(os.path.realpath(''), 'data')
CATEGORY_NAMES = ["bear", "elephant", "leopard", "zebra"]
NUM_CATEGORIES = len(CATEGORY_NAMES)

# Detect if we have a GPU available
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fprint("Running on {}...".format(args.device), args, True)

# All pre-trained models expect input images normalized in the same way
# For details: https://pytorch.org/docs/master/torchvision/models.html
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Data augmentation and normalization for training
# Just resizing and normalization for val and test
trans_dict = {
    'train': transforms.Compose([transforms.RandomResizedCrop(INPUT_SIZE),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 NORMALIZE
                                ]),

    'val': transforms.Compose([transforms.Resize(INPUT_SIZE),
                               transforms.CenterCrop(INPUT_SIZE),
                               transforms.ToTensor(),
                               NORMALIZE
                              ]),

    'test': transforms.Compose([transforms.Resize(INPUT_SIZE),
                                transforms.CenterCrop(INPUT_SIZE),
                                transforms.ToTensor(),
                                NORMALIZE
                               ])
}

# Create training, validation and test datasets
img_folders = {x: IF(os.path.join(DATA_DIR, 'prepared', x), trans_dict[x])
               for x in ['train', 'val', 'test']}

# Create training, validation and test dataloaders
# When using CUDA, set num_workers=1 and pin_memory=True
data_loaders = {x: DL(img_folders[x], batch_size=args.batch_size, shuffle=True,
                      num_workers=int(args.device == 'cuda'),
                      pin_memory=(args.device == 'cuda'))
                for x in ['train', 'val', 'test']}

if args.show_plots:
    display_single(img_folders, NORMALIZE, CATEGORY_NAMES, img_id=153)  # 153 is arbitrarily chosen
    display_multiple(args, NORMALIZE, CATEGORY_NAMES, data_loaders, N=4)  # display 4 images

# Initialize model
model, params_to_update = initialize_model(NUM_CATEGORIES, args)

# Send the model to CPU or GPU
model = model.to(args.device)

# Setup the optimizer
optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)

# Setup the loss function
criterion = torch.nn.CrossEntropyLoss()

# Log training parameters
TRAIN_PARAMS = ""
for param in vars(args):
    TRAIN_PARAMS += param + '=' + str(getattr(args, param)) + '\n'
fprint("\nTRAINING WITH PARAMS:\n{}".format(TRAIN_PARAMS), args, True)

# Train and evaluate
model, optimizer = train_model(model, data_loaders, criterion, optimizer, args)

# Or load a checkpoint
# checkpoint = torch.load('models/alexnet_pretrained_0.995536.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# acc = checkpoint['acc']

test_score = test_model(model, data_loaders, args)

# Prediction for a single test image
# print(predict(model, NORMALIZE, CATEGORY_NAMES, INPUT_SIZE, 'data/bear_test.JPEG', args))
