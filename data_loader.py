import os
from torchvision import transforms
from torchvision.datasets import ImageFolder as IF
from torch.utils.data import DataLoader as DL

from utils import display_single, display_multiple

def get_data_loaders(data_dir, category_names, args):
    # All pre-trained models expect input images normalized in the same way
    # For details: https://pytorch.org/docs/master/torchvision/models.html
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Data augmentation and normalization for training
    # Just resizing and normalization for val and test
    trans_dict = {
        'train': transforms.Compose([transforms.RandomResizedCrop(args.input_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize
                                    ]),

        'val': transforms.Compose([transforms.Resize(args.input_size),
                                transforms.CenterCrop(args.input_size),
                                transforms.ToTensor(),
                                normalize
                                ]),

        'test': transforms.Compose([transforms.Resize(args.input_size),
                                    transforms.CenterCrop(args.input_size),
                                    transforms.ToTensor(),
                                    normalize
                                ])
    }

    # Create training, validation and test datasets
    img_folders = {x: IF(os.path.join(data_dir, 'prepared', x), trans_dict[x])
                for x in ['train', 'val', 'test']}

    # Create training, validation and test dataloaders
    # When using CUDA, set num_workers=1 and pin_memory=True
    data_loaders = {x: DL(img_folders[x], batch_size=args.batch_size, shuffle=True,
                        num_workers=int(args.device == 'cuda'),
                        pin_memory=(args.device == 'cuda'))
                    for x in ['train', 'val', 'test']}

    if args.debug:
        display_single(img_folders, normalize, category_names, img_id=153)  # 153 is arbitrarily chosen
        display_multiple(args, normalize, category_names, data_loaders, N=4)  # display 4 images

    return data_loaders
