from backet_dataset import BacketDataset
from torch.utils.data import DataLoader
import torch
from model import CNN
import torch.optim as optim
from torchvision import datasets, transforms

import os


def initialize(args):
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    model = select_model(args)

    optimizer = select_optimizer(args,model)

    if args.cuda:
        model.to(args.device)

    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    train_params = {'num_workers': 2, 'batch_size': args.batch_size,'shuffle': True}
    valid_params = {'num_workers': 2, 'batch_size': args.batch_size, 'shuffle': True}

    train_generator = datasets.ImageFolder(args.root_path, train_transforms)
    train, val,test = torch.utils.data.random_split(train_generator, [48000, 12000,10000],)

    train_loader = DataLoader(train, pin_memory=True, **train_params)
    valid_loader = DataLoader(val, pin_memory=True, **valid_params)
    test_loader = DataLoader(test, pin_memory=True, **valid_params)

    return train_loader, valid_loader, model, optimizer


def select_model(args):
    return CNN(args.classes, args.model)


def select_optimizer(args, model):
    if args.opt == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum = 0.5, weight_decay =args.weight_decay)

    elif args.opt == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay =args.weight_decay)

    elif args.opt == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    else:
        assert False, "unspecified optimizer"


