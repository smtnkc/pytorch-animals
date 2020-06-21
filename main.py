from __future__ import division
from __future__ import print_function

import platform
import json
import torch
import torch.optim as optim
import torchvision

from data_loader import get_data_loaders
from utils import fprint, load_args, export_args, load_json_args, init_random_seeds
from model_helper import initialize_model, train_model, test_model
from plot import generate_plots


def main():

    args = load_args()
    init_random_seeds(args.seed)

    # EXPORT ARGS AS JSON
    json_path = export_args(args)  # write to file
    json_args = load_json_args(json_path)  # read from file
    fprint("RUNNING ARGS:\n{}\n".format(json.dumps(json_args, indent=4)), args)

    fprint("Python Version: {}".format(platform.python_version()), args)
    fprint("PyTorch Version: {}".format(torch.__version__), args)
    fprint("Torchvision Version: {}".format(torchvision.__version__.split('a')[0]), args)

    # Get data loaders
    data_loaders = get_data_loaders(args)

    # Initialize model
    model, params_to_update = initialize_model(is_pretrained=args.pretrained)

    fprint("\nARCHITECTURE:\n{}\n".format(model), args)

    for name, param in model.named_parameters():
        fprint("{:25} requires_grad = {}".format(name, param.requires_grad), args)

    # Send the model to CPU or GPU
    model = model.to(torch.device(args.device))

    # Setup the optimizer
    if args.optimizer == 'sgdm':
        optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.AdamW(params_to_update, lr=args.lr)

    # Setup the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Train and evaluate
    model, optimizer = train_model(model, data_loaders, criterion, optimizer, args)

    # Test
    test_model(model, data_loaders, args)

    # Generate plots:
    generate_plots(json_path)


if __name__ == "__main__":
    main()
