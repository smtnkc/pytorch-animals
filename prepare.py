"""
This function creates the directory structure as shown below.
This structure is required by torchvision.datasets.ImageFolder().
"""

import os
import argparse
import shutil
from sklearn.model_selection import train_test_split

import cfg

def prepare_image_folders(args):

    """
    Prepares directory structure given below.
    data/
       prepared/          -> 1400
           train/         -> 896 total
              bear/       -> 224
              elephant/   -> 224
              leopard/    -> 224
              zebra/      -> 224
          test/           -> 280 total
              bear/       -> 70
              elephant/   -> 70
              leopard/    -> 70
              zebra/      -> 70
          val/            -> 224 total
              bear/       -> 56
              elephant/   -> 56
              leopard/    -> 56
              zebra/      -> 56
    """

    prep_path = os.path.join(cfg.DATA_DIR, 'prepared')

    # create/clean train, val and test folders
    folders = ['train', 'val', 'test']
    for folder in folders:
        folder_path = os.path.join(prep_path, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # remove if already exists
        for category in cfg.CATEGORIES:
            os.makedirs(os.path.join(folder_path, category))

    for category in cfg.CATEGORIES:
        category_path = os.path.join(cfg.DATA_DIR, 'animals', category)
        category_file_names = []

        # get paths of files
        for file_name in os.listdir(category_path):
            category_file_names.append(file_name)

        # split paths into train, val and test
        train_file_names, test_file_names = train_test_split(category_file_names,
                                                             test_size=0.2,
                                                             random_state=args.seed)
        train_file_names, val_file_names = train_test_split(train_file_names,
                                                            test_size=0.2,
                                                            random_state=args.seed)

        print("{:<10s} -> Divided into {} train / {} val / {} test"
              .format(category, len(train_file_names), len(val_file_names), len(test_file_names)),
              end='')

        # copy train files
        for file_name in train_file_names:
            src = os.path.join(category_path, file_name)
            dst = os.path.join(prep_path, 'train', category, file_name)
            shutil.copyfile(src, dst)

        # copy val files
        for file_name in val_file_names:
            src = os.path.join(category_path, file_name)
            dst = os.path.join(prep_path, 'val', category, file_name)
            shutil.copyfile(src, dst)

        # copy test files
        for file_name in test_file_names:
            src = os.path.join(category_path, file_name)
            dst = os.path.join(prep_path, 'test', category, file_name)
            shutil.copyfile(src, dst)

        print("   -> Copied!")


def main():
    parser = argparse.ArgumentParser(description='PyTorch Animals Prepare')
    parser.add_argument('--seed', default=cfg.SEED, type=int)
    args = parser.parse_args()
    prepare_image_folders(args)


if __name__ == "__main__":
    main()
