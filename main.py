import torch
import argparse
import wandb
from initialize import initialize
from save_model import save_model
from train import train
from evaluation import evaluation


def main():
    args = get_args()
    wandb.init()
    wandb.config.update(args)
    torch.backends.cudnn.benchmark = True
    [train_generator, valid_generator, model, optimizer] = initialize(args)

    wandb.watch(model)
    for epoch in range(1, args.epochs_number + 1):
        train(args, model, train_generator, epoch, optimizer)
        evaluation(args, model, valid_generator)

        save_model(model, optimizer, args, loss, epoch)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--dataset_name', type=str, default='shape_class', help='dataset_name')
    parser.add_argument('--epochs_number', type=int, default=64, help='epoch number for training')
    parser.add_argument('--classes', type=int, default=10, help='dataset_classes')
    parser.add_argument('--lr', type=int, default=0.01, help='learning rate')
    parser.add_argument('--device', type=int, default=0, help='gpu device')  # default is 0
    parser.add_argument('--weight_decay', type=int, default=1e-7, help='dropout')
    parser.add_argument('--cuda', action='store_true', default=True, help='gpu for training acceleration')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd','adam','rmsprop'))
    parser.add_argument('--root_path', type=str, default='./data', help='path to dataset')
    parser.add_argument('--save', type=str, default='/save/backetnet', help='path to checkpoint save directory')

    args = parser.parse_args()
    return args

main()