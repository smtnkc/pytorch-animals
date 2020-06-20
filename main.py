from __future__ import division
from __future__ import print_function

import os
import platform
import json
import torch
import torch.optim as optim
import torchvision

from data_loader import get_data_loaders
from utils import fprint, load_args, create_config_file
from model_helper import initialize_model, train_model, test_model, predict
from plot import generate_plots

def main():

    args = load_args()

    fprint("Python Version: {}".format(platform.python_version()), args)
    fprint("PyTorch Version: {}".format(torch.__version__), args)
    fprint("Torchvision Version: {}".format(torchvision.__version__.split('a')[0]), args)
    fprint("Running on {}...".format(args.device), args)

    data_dir = os.path.join(os.path.realpath(''), 'data')
    category_names = ["bear", "elephant", "leopard", "zebra"]
    num_categories = len(category_names)

    data_loaders = get_data_loaders(data_dir, category_names, args)

    # Initialize model
    model, params_to_update = initialize_model(num_categories, args)

    # Send the model to CPU or GPU
    model = model.to(torch.device(args.device))

    # Setup the optimizer
    if args.optimizer == 'sgdm':
        optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.AdamW(params_to_update, lr=args.lr)

    # Setup the loss function
    criterion = torch.nn.CrossEntropyLoss()

    config_path = create_config_file(args)
    with open(config_path, 'r') as json_file:
        configs = json.load(json_file)

    fprint("\nTRAINING PARAMS:\n{}".format(json.dumps(configs, indent=4)), args)

    # Train and evaluate
    model, optimizer = train_model(model, data_loaders, criterion, optimizer, args)

    # Test
    test_model(model, data_loaders, args)

    # Generate plots:
    generate_plots(config_path)


if __name__ == "__main__":
    main()
