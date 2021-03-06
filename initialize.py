from backet_dataset import BacketDataset
from torch.utils.data import DataLoader
import torch
from torch import nn
from model import CNN, CNN1, Lenet5, FullyConnected
import torch.optim as optim
from torchvision import datasets, transforms
from utils import load_checkpoint
import os


def initialize(args, loaded_model):
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    if not loaded_model:
        model = select_model(args)
        print(model)
        optimizer = select_optimizer(args, model)

    else:
        model = select_model(args)

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(512, 10)

        model_state_dict, optimizer_state_dict = load_checkpoint(args.save + '/' + 'backet_net.pt')

        model.load_state_dict(model_state_dict)

        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)

        optimizer = optim.Adam(params_to_update, args.lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(optimizer_state_dict)

    if args.cuda:
        model.to(args.device)

    train_transforms = get_transforms(args)

    train_params = {'num_workers': 2, 'batch_size': args.batch_size, 'shuffle': True}
    valid_params = {'num_workers': 2, 'batch_size': args.batch_size, 'shuffle': True}

    train_generator = datasets.ImageFolder(args.root_path + '/' + 'train', train_transforms)
    train, val = torch.utils.data.random_split(train_generator, [48000, 12000])

    train_loader = DataLoader(train, pin_memory=True, **train_params)
    valid_loader = DataLoader(val, pin_memory=True, **valid_params)

    return train_loader, valid_loader, model, optimizer


def select_model(args):
    if args.model == "resnet18":
        return CNN(args.classes)

    elif args.model == "resnet34":
        return CNN1(args.classes)

    elif args.model == "Lenet5":
        return Lenet5(args.classes)

    elif args.model == "FullyConnected":
        return FullyConnected(args.classes)

    else:
        assert False, "unspecified model"


def get_transforms(args):
    if args.model == "resnet18":
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.2858, 0.2858, 0.2858], [0.2869, 0.2869, 0.2869])
        ])

    elif args.model == "resnet34":
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.2858, 0.2858, 0.2858], [0.2869, 0.2869, 0.2869])
        ])

    elif args.model == "Lenet5":
        train_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    elif args.model == "FullyConnected":
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    else:
        assert False, "unspecified model"

    return train_transforms


def select_optimizer(args, model):
    if args.opt == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay =args.weight_decay)

    elif args.opt == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.opt == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.opt == 'adamw':
        return torch.optim.AdamW(model.parameters(), betas=(0.9, 0.99), eps=1e-5, weight_decay=1e-2)
    else:
        assert False, "unspecified optimizer"


