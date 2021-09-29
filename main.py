import torch
from utils import get_args
import wandb
from initialize import initialize
from train import train
from evaluation import evaluation
from utils import RunningAverage
from utils import save_checkpoint
import os


def main():
    args = get_args()
    wandb.init()
    wandb.config.update(args)
    torch.backends.cudnn.benchmark = True

    loaded_model = False

    [train_loader, valid_loader, model, optimizer] = initialize(args, loaded_model)
    scaler = torch.cuda.amp.GradScaler()

    wandb.watch(model)
    best_acc = 0
    run_avg = RunningAverage()

    if args.opt == 'adamw':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1)
    else:
        torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, cyclic_momentum=False)

    for epoch in range(1, args.epochs_number + 1):
        run_avg.reset_train()
        run_avg.reset_val()
        scheduler.step()

        train(args, model, train_loader, epoch, optimizer, scaler, run_avg)
        val_acc = evaluation(args, model, valid_loader, epoch, run_avg)

        if best_acc < val_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, args, epoch)


main()