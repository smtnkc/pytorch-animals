import time
import copy
import torch
import pandas as pd
from torchvision import models
from torchvision import transforms
from PIL import Image

from utils import fprint, calculate_metrics

def initialize_model(out_size, args):

    model = models.alexnet(pretrained=args.use_pretrained)

    # initially disable all parameter updates
    if args.use_pretrained:
        for param in model.parameters():
            param.requires_grad = False

    # reshape the output layer
    in_size = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(in_size, out_size)

    if args.use_pretrained:
        params_to_update = []
        for param in model.parameters():
            if param.requires_grad:
                params_to_update.append(param)  # parameters of reshaped layer
    else:
        params_to_update = model.parameters()  # parameters of all layers

    fprint("\nARCHITECTURE:", args, True)
    for name, param in model.named_parameters():
        fprint("{:25} requires_grad = {}".format(name, param.requires_grad), args, True)

    return model, params_to_update

#
#
#
#
#
#


def train_model(model, data_loaders, criterion, optimizer, args):
    stats_df = pd.DataFrame(columns=['train_loss', 'train_acc', 'train_f1',
                                     'val_loss', 'val_acc', 'val_f1'])

    since = time.time()

    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_opt_state_dict = copy.deepcopy(optimizer.state_dict())
    best_loss = 999999.9
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        fprint('\nEpoch {}/{}'.format(epoch, args.epochs - 1), args, True)
        fprint('-' * 10, args, True)

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
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

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

                # statistics
                batch_loss = loss.item() * inputs.size(0)
                batch_correct = torch.sum(preds == labels.data)
                phase_loss += batch_loss
                phase_corrects += batch_correct
                phase_preds = torch.cat((phase_preds, preds), 0)
                phase_labels = torch.cat((phase_labels, labels), 0)

            epoch_loss = phase_loss / len(data_loaders[phase].dataset)
            epoch_acc, epoch_f1 = calculate_metrics(phase_preds, phase_labels)

            fprint('{:7}  loss: {:.6f}   acc: {:.6f}   f1: {:.6f}'.format(
                phase, epoch_loss, epoch_acc, epoch_f1), args, True)

            stats_df.at[epoch, phase + '_loss'] = round(epoch_loss, 6)
            stats_df.at[epoch, phase + '_acc'] = round(epoch_acc, 6)
            stats_df.at[epoch, phase + '_f1'] = round(epoch_f1, 6)

            # define the new bests
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_state_dict = copy.deepcopy(model.state_dict())
                best_opt_state_dict = copy.deepcopy(optimizer.state_dict())
                best_loss = copy.deepcopy(epoch_loss)
                best_epoch = epoch

    time_elapsed = time.time() - since
    fprint('\nTraining completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), args, True)

    # reload best model weights and best optimizer variables
    model.load_state_dict(best_model_state_dict)
    optimizer.load_state_dict(best_opt_state_dict)

    # save best checkpoint
    if args.use_pretrained:
        cp_path = './models/pretrained_acc_{:.6f}_{}.pth'.format(best_acc, args.t_start)
        stats_path = './logs/stats_pretrained_{}.csv'.format(args.t_start)
    else:
        cp_path = './models/scratch_acc_{:.6f}_{}.pth'.format(best_acc, args.t_start)
        stats_path = './logs/stats_scratch_{}.csv'.format(args.t_start)

    stats_df.to_csv(stats_path, sep=',', index=False)  # write loss and acc values
    fprint('Saved loss and accuracy stats -> {}'.format(stats_path), args, True)

    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_state_dict,
        'optimizer_state_dict': best_opt_state_dict,
        'loss': best_loss,
        'acc': best_acc
    }, cp_path)

    fprint('Saved best checkpoint ->  Epoch: {} Val Loss: {:.6f} Val Acc: {:6f}'
          .format(best_epoch, best_loss, best_acc), args, True)

    return model, optimizer

#
#
#
#
#
#


def test_model(model, data_loaders, args):
    was_training = model.training  # store mode
    model.eval()  # run in evaluation mode

    with torch.no_grad():
        running_corrects = 0
        for inputs, labels in data_loaders['test']:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            batch_correct = torch.sum(preds == labels.data)
            running_corrects += batch_correct

        dataset = data_loaders['test'].dataset
        acc = running_corrects.double() / len(dataset)

        fprint('\nTEST RESULTS:\n{}/{} predictions are correct -> Test Acc: {:.4f}'
               .format(running_corrects, len(dataset), acc), args, True)

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
        inp = img.to(args.device)
        output = model(inp)
        _, pred = torch.max(output, 1)

    model.train(mode=was_training)  # reinstate the previous mode

    return category_names[pred.numpy()[0]]
