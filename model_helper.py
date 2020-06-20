import os
import time
import copy
import torch
import pandas as pd
from torchvision import models
from torchvision import transforms
from PIL import Image

from utils import fprint, calculate_metrics

def initialize_model(out_size, args):

    model = models.alexnet(pretrained=args.pretrained)

    # initially disable all parameter updates
    if args.pretrained:
        for param in model.parameters():
            param.requires_grad = False

    # reshape the output layer
    in_size = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(in_size, out_size)

    if args.pretrained:
        params_to_update = []
        for param in model.parameters():
            if param.requires_grad:
                params_to_update.append(param)  # parameters of reshaped layer
    else:
        params_to_update = model.parameters()  # parameters of all layers

    fprint("\nARCHITECTURE:\n\n{}\n".format(model), args)

    for name, param in model.named_parameters():
        fprint("{:25} requires_grad = {}".format(name, param.requires_grad), args)

    return model, params_to_update

#
#
#
#
#
#


def train_model(model, data_loaders, criterion, optimizer, args):

    # create states df and csv file
    stats_df = pd.DataFrame(
        columns=['epoch', 'train_loss', 'train_acc', 'train_f1', 'val_loss', 'val_acc', 'val_f1'])
    stats_path = 'logs/{}_{}.csv'.format('pt' if args.pretrained else 'fs', args.t_start)
    stats_df.to_csv(stats_path, sep=',', index=False)  # write loss and acc values
    fprint('\nCreated stats file\t-> {}'.format(stats_path), args)
    fprint('\nTRAINING {} EPOCHS...\n'.format(args.epochs), args)

    since = time.time()

    # initialize best values
    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_opt_state_dict = copy.deepcopy(optimizer.state_dict())
    best_loss = 999999.9
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            phase_loss = 0.0
            phase_corrects = 0
            phase_preds = torch.LongTensor()
            phase_labels = torch.LongTensor()

            # Iterate over data
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(torch.device(args.device))
                labels = labels.to(torch.device(args.device))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # stats
                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data)
                phase_loss += batch_loss
                phase_corrects += batch_corrects
                phase_preds = torch.cat((phase_preds, preds), 0)
                phase_labels = torch.cat((phase_labels, labels), 0)

            epoch_loss = phase_loss / len(data_loaders[phase].dataset)
            epoch_acc, epoch_f1 = calculate_metrics(phase_preds, phase_labels)

            stats_df.at[0, 'epoch'] = epoch
            stats_df.at[0, phase + '_loss'] = round(epoch_loss, 6)
            stats_df.at[0, phase + '_acc'] = round(epoch_acc, 6)
            stats_df.at[0, phase + '_f1'] = round(epoch_f1, 6)

            # define the new bests
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_state_dict = copy.deepcopy(model.state_dict())
                best_opt_state_dict = copy.deepcopy(optimizer.state_dict())
                best_loss = copy.deepcopy(epoch_loss)
                best_epoch = epoch

        # append epoch stats to file
        fprint(stats_df.to_string(index=False, header=(epoch == 0), col_space=10, justify='right'), args)
        stats_df.to_csv(stats_path, mode='a', header=False, index=False)

    time_elapsed = time.time() - since
    fprint('\nTraining completed in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60), args)

    # reload best model weights and best optimizer variables
    model.load_state_dict(best_model_state_dict)
    optimizer.load_state_dict(best_opt_state_dict)

    # save best checkpoint
    cp_dir = 'models'
    if not os.path.exists(cp_dir):
        os.makedirs(cp_dir)

    cp_path = os.path.join(cp_dir, '{}_{}_{:.6f}.pth'.format(
        'pt' if args.pretrained else 'fs', args.t_start, best_acc))

    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_state_dict,
        'optimizer_state_dict': best_opt_state_dict,
        'loss': best_loss,
        'acc': best_acc
    }, cp_path)

    fprint('Saved best checkpoint\t-> {}'.format(cp_path), args)

    return model, optimizer

#
#
#
#
#
#


def test_model(model, data_loaders, args):
    fprint('\nTESTING...', args)
    was_training = model.training  # store mode
    model.eval()  # run in evaluation mode

    with torch.no_grad():
        phase_corrects = 0
        phase_preds = torch.LongTensor()
        phase_labels = torch.LongTensor()

        for inputs, labels in data_loaders['test']:
            inputs = inputs.to(torch.device(args.device))
            labels = labels.to(torch.device(args.device))

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            batch_corrects = torch.sum(preds == labels.data)
            phase_corrects += batch_corrects
            phase_preds = torch.cat((phase_preds, preds), 0)
            phase_labels = torch.cat((phase_labels, labels), 0)

        dataset = data_loaders['test'].dataset
        acc, f1 = calculate_metrics(phase_preds, phase_labels)

        fprint('{}/{} predictions are correct -> Test acc: {:.6f}   f1: {:.6f}\n'.format(
            phase_corrects, len(dataset), acc, f1), args)

    model.train(mode=was_training)  # reinstate the previous mode

    return acc

#
#
#
#
#
#

def predict(model, normalize, category_names, input_size, img_path, args):
    """ Prediction for a single test image """
    was_training = model.training  # store mode
    model.eval()  # run in evaluation mode

    loader = transforms.Compose([transforms.Resize(input_size),
                                 transforms.CenterCrop(input_size),
                                 transforms.ToTensor(),
                                 normalize
                                ])

    img = Image.open(img_path)
    img = loader(img).float()
    img = img.unsqueeze(0)

    with torch.no_grad():
        inp = img.to(torch.device(args.device))
        output = model(inp)
        _, pred = torch.max(output, 1)

    model.train(mode=was_training)  # reinstate the previous mode

    return category_names[pred.numpy()[0]]
