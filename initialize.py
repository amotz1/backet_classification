from backet_dataset import BacketDataset
from torch.utils.data import DataLoader
import torch
from model import CNN
import torch.optim as optim
import os


def initialize(args):
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    model = select_model(args)

    optimizer = select_optimizer(args,model)

    if args.cuda:
        model.to(args.device)

    train_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 2}
    valid_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 2}

    train_generator = DataLoader(
        datasets.MNIST(root=args.rootpath, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **train_params)

    valid_generator = DataLoader(
        datasets.MNIST(root=args.dataroot, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **valid_params)

    # train_generator = torch.DataLoader(train_loader, pin_memory=True, **train_params)
    # valid_generator = torch.DataLoader(valid_loader, pin_memory=True, **valid_params)

    return optimizer, model, train_generator, valid_generator


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


