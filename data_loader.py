import os
from torchvision.datasets import ImageFolder as IF
from torch.utils.data import DataLoader as DL
from torchvision import transforms

import cfg
from utils import display_single, display_multiple

def get_data_loaders(args):
    # Data augmentation and normalization for training
    # Just resizing and normalization for val and test
    trans_dict = {
        'train': transforms.Compose([transforms.RandomResizedCrop(args.input_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     cfg.NORMALIZE
                                    ]),

        'val': transforms.Compose([transforms.Resize(args.input_size),
                                   transforms.CenterCrop(args.input_size),
                                   transforms.ToTensor(),
                                   cfg.NORMALIZE
                                  ]),

        'test': transforms.Compose([transforms.Resize(args.input_size),
                                    transforms.CenterCrop(args.input_size),
                                    transforms.ToTensor(),
                                    cfg.NORMALIZE
                                   ])
    }

    # Create training, validation and test datasets
    img_folders = {x: IF(os.path.join(cfg.DATA_DIR, 'prepared', x), trans_dict[x])
                   for x in cfg.PHASES}

    # Create training, validation and test dataloaders
    # When using CUDA, set num_workers=1 and pin_memory=True
    data_loaders = {x: DL(img_folders[x], batch_size=args.batch_size, shuffle=True,
                          num_workers=int(args.device == 'cuda'),
                          pin_memory=(args.device == 'cuda'))
                    for x in cfg.PHASES}

    if args.display_images:
        display_single(args, img_folders, img_id=153)  # 153 is arbitrarily chosen
        display_multiple(args, data_loaders, N=4)  # display 4 images

    return data_loaders
