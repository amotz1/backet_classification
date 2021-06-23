from backet_dataset import BacketDataset
from torch.utils.data import DataLoader
import torch
from model import CNN
import torch.optim as optim
from torchvision import datasets, transforms
from utils import load_checkpoint
import os


def initialize(args, loaded_model):
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    if not loaded_model:
        model = select_model(args)
        optimizer = select_optimizer(args, model)

    else:
        model, optimizer = load_checkpoint(args.savepath + '/' + 'backet_net.pt')

        for param in model.parameters():
            param.requires_grad = False

        print(model)

    if args.cuda:
        model.to(args.device)

    if loaded_model is False:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

    else:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
        ])

    train_params = {'num_workers': 2, 'batch_size': args.batch_size,'shuffle': True}
    valid_params = {'num_workers': 2, 'batch_size': args.batch_size, 'shuffle': True}

    train_generator = datasets.ImageFolder(args.root_path + '/' + 'train', train_transforms)
    train, val = torch.utils.data.random_split(train_generator, [48000, 12000])

    train_loader = DataLoader(train, pin_memory=True, **train_params)
    valid_loader = DataLoader(val, pin_memory=True, **valid_params)

    return train_loader, valid_loader, model, optimizer


def select_model(args):
    return CNN(args.classes, args.model)


def select_optimizer(args, model):
    if args.opt == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay =args.weight_decay)

    elif args.opt == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.opt == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        assert False, "unspecified optimizer"


