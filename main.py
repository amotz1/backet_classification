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
    parser.add_argument('--batch size', type=int, defult=64, help='batch size for training')
    parser.add_argument('--dataset_name', type=str, defult='shape_class', help='dataset_name')
    parser.add_argument('--epochs_number', type=int, defult=64, help='epoch number for training')
    parser.add_argument('--classes', type=int, defult=3, help='dataset_classes')
    parser.add_argument('--lr', type=int, defult=0.01, help='learning rate')
    parser.add_argument('--device', type=int, default=0, help='gpu device')  # default is 0
    parser.add_argument('--weight_decay', type=int, defult=1e-7, help='dropout')
    parser.add_argument('--cuda', action='store_true', defult=True, help='gpu for training acceleration')
    parser.add_argument('--model', type=str, defult='resnet50')
    parser.add_argument('--opt', type=str, defult='adam', choices=('sgd','adam','rmsprop'))
    parser.add_argument('--root_path', type=str, defult='./data', help='path to dataset')
    parser.add_argument('--save', type=str, defult='/save/backetnet', help='path to checkpoint save directory')

    args = parser.parse_args()
    return args

main()