
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import glob

#Uses Code from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

def get_model_name(opts):
    return os.path.basename(opts.dataroot) + '_lr' + str(opts.lr) + '_bs' + str(opts.batch_size)


def prepare_model(model, num_classes, freeze_layers=True):
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

def create_model(opts):
    device = torch.device('cuda:{}'.format(opts.gpu_ids[0])) if opts.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
    if opts.phase == 'evaluate':
        state_dict = torch.load(os.path.join(opts.model_directory, opts.model_name + '.pth'), map_location=device)
        model = torchvision.models.resnet50(pretrained=False, progress=True)
        prepare_model(model, num_classes=opts.num_classes, freeze_layers=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
    else:
        model_path = os.path.join(opts.model_directory, get_model_name(opts) + '.pth')
        # load pretrained resnet model
        model = torchvision.models.resnet50(pretrained=True, progress=True)
        prepare_model(model, num_classes=opts.num_classes, freeze_layers=True)
        model = model.to(device)
        train_model(model, opts, device=device, invert_train_val=True)
        
        if len(opts.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(model.cpu().state_dict(), model_path)
            model.cuda(opts.gpu_ids[0])
        else:
            torch.save(model.cpu().state_dict(), model_path)
    return model

def train_model(model, opts, device, invert_train_val=True):    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=opts.lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    phase = opts.phase
    
    data_dir = opts.dataroot
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opts.batch_size,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    optimize_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, device, dataset_sizes, opts)

def optimize_model(model, dataloader, criterion, optimizer, scheduler, device, dataset_sizes, opts, num_epochs=100):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    results = []
    results_path = get_model_name(opts)
    results_path = os.path.join(opts.model_directory, results_path + '.txt')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            results.append('{},{},{:.4f},{:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    with open(results_path, 'w') as f:
        for res in results:
            f.write(res + '\n')
    model.load_state_dict(best_model_wts)
    return model
