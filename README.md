# Pytorch Animals
Multi-label image classification with transfer learning.

# Using the Dataset:
1. Download and extract **[animals.zip](https://drive.google.com/file/d/1U1xhb5ZmyPP2jORd1TVfJz1IEdg9JY4I/view?usp=sharing)** file (194.5 Mb).
2. Move the extracted ``animals`` folder into ``data/`` directory.

The final directory structure should look like:

<img width="158" alt="directory_structure" src="https://user-images.githubusercontent.com/25348698/84280950-a738de00-ab40-11ea-92e0-66f58b95f612.png">

# Running Instructions:
Before the first run, prepare the image folders using:

```bash
python3 prepare.py
```

Then run like:

```bash
python3 main.py --epoch=3
```

To plot results:

```bash
python3 plot.py --json_path='dumps/xxx/args.json'
```

To predict an image using a checkpoint:

```bash
python3 predict.py --model_path='models/checkpoint_file.pth' --img_path='test_image_file.JPEG' --label='bear'
```

# References:
* Pre-trained models: https://pytorch.org/docs/master/torchvision/models.html
* Tutorial 1: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
* Tutorial 2: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
* Tutorial 3: https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce
* YouTube Playlist: https://www.youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
* Data Loader Example: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py
* Optimizer state-dict: https://discuss.pytorch.org/t/importance-of-optimizers-when-continuing-training/64788
