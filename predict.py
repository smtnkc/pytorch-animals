import os
import argparse
import json
import torch
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cfg
from utils import load_json_args
from model_helper import initialize_model


def plot_prediction(img_path, model_path, title):
    img = mpimg.imread(img_path)
    plt.figure(figsize=(8, 5))
    plt.imshow(img)
    plt.title(title)
    plot_path = model_path[:-4] + '.png'
    plt.savefig(plot_path, bbox_inches='tight', transparent="True", pad_inches=0.1)
    plt.close()


# Load a checkpoint
def predict(model, img_path, json_args):
    """ Prediction for a single test image """
    was_training = model.training  # store mode
    model.eval()  # run in evaluation mode

    loader = transforms.Compose([transforms.Resize(json_args['input_size']),
                                 transforms.CenterCrop(json_args['input_size']),
                                 transforms.ToTensor(),
                                 cfg.NORMALIZE
                                ])

    img = Image.open(img_path)
    img = loader(img).float()
    img = img.unsqueeze(0)

    with torch.no_grad():
        inp = img.to(torch.device(json_args['device']))
        output = model(inp)
        sm = torch.nn.Softmax(dim=1)
        probabilities = sm(output).numpy()[0]
        print('\nPrediction probabilities -> {}'.format(probabilities))
        _, pred = torch.max(output, 1)
        pred_category_id = pred.numpy()[0]
        pred_probability = probabilities[pred_category_id]

    model.train(mode=was_training)  # reinstate the previous mode

    return pred_category_id, pred_probability

def main():
    parser = argparse.ArgumentParser(description='PyTorch Animals Predict')
    parser.add_argument('--model_path', default='', type=str)
    parser.add_argument('--img_path', default='', type=str)
    args = parser.parse_args()

    sub_dump_dir = os.path.join(cfg.DUMP_DIR, os.path.basename(args.model_path)[:-13])
    json_path = os.path.join(sub_dump_dir, 'args.json')

    json_args = load_json_args(json_path)

    if json_args is None:
        return

    print("\nRUNNING ARGS:\n{}\n".format(json.dumps(json_args, indent=4)))

    # Initialize model
    model, params_to_update = initialize_model(is_pretrained=json_args['pretrained'])

    # Send the model to CPU or GPU
    model = model.to(torch.device(json_args['device']))

    # Setup the optimizer
    if json_args['optimizer'] == 'sgdm':
        optimizer = optim.SGD(params_to_update, lr=json_args['lr'],
                              weight_decay=json_args['weight_decay'], momentum=0.9)
    elif json_args['optimizer'] == 'adam':
        optimizer = optim.AdamW(params_to_update, lr=json_args['lr'],
                                weight_decay=json_args['weight_decay'])

    checkpoint = torch.load(args.model_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(model)

    print("\nCheckpoint loaded -> epoch: {} / val loss: {:.6f} / val acc:{:.6f}".format(
        checkpoint['epoch'], checkpoint['loss'], checkpoint['acc']
    ))

    pred_category_id, pred_prob = predict(model, args.img_path, json_args)
    result_text = '{} ({:.2f}%)'.format(cfg.CATEGORIES[pred_category_id], pred_prob*100)
    print('\nPrediction  ->', result_text)
    plot_prediction(args.img_path, args.model_path, result_text)

if __name__ == "__main__":
    main()
