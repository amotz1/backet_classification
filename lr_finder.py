from model import Lenet5, FullyConnected, CNN
from torch_lr_finder.lr_finder import LRFinder
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--dataset_name', type=str, default='shape_class', help='dataset_name')
    parser.add_argument('--epochs_number', type=int, default=8, help='epoch number for training')
    parser.add_argument('--classes', type=int, default=10, help='dataset_classes')
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
    parser.add_argument('--device', type=int, default=0, help='gpu device')  # default is 0
    parser.add_argument('--weight_decay', type=int, default=1e-7, help='dropout')
    parser.add_argument('--cuda', action='store_true', default=True, help='gpu for training acceleration')
    parser.add_argument('--model', type=str, default='Lenet5')
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd','adam','rmsprop'))
    parser.add_argument('--root_path', type=str, default='/content/MnistData', help='path to dataset')
    parser.add_argument('--save', type=str, default='/content/save/backetnet', help='path to checkpoint save directory')

    args = parser.parse_args()
    return args


lf()
