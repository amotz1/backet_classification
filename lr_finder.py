from model import Lenet5, FullyConnected, CNN
from torch_lr_finder.lr_finder import LRFinder
from utils import get_args
from initialize import select_model, select_optimizer, get_transforms
import argparse


def lf():
    args = get_args()
    model = select_model(args)
    optimizer = select_optimizer(args, model)
    train_transforms = get_transforms(args)

    train_params = {'num_workers': 2, 'batch_size': args.batch_size, 'shuffle': True}

    train_generator = datasets.ImageFolder(args.root_path + '/' + 'train', train_transforms)
    train, _ = torch.utils.data.random_split(train_generator, [48000, 12000])

    train_loader = DataLoader(train, pin_memory=True, **train_params)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode="exp")

lf()
