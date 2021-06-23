import torch
from torch import nn
import time
from utils import RunningAverage
from utils import acc
import wandb


def train(args, model, train_loader, epoch, optimizer, scaler, run_avg):
    model.train()
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss(reduction='mean')

    for batch_index, input_tensor in enumerate(train_loader):
        input_data, target = input_tensor

        if args.cuda:
            input_data = input_data.to(device)
            target = target.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(input_data)
            loss = criterion(output, target)

        run_avg.update_train_loss_avg(loss.item(), args.batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        accuracy = acc(output, target)
        run_avg.update_train_acc_avg(accuracy, args.batch_size)

        if batch_index % 10 == 9:
            print('epoch =', epoch, ' train_loss = ', run_avg.train_loss_run_avg, ' accuracy =', run_avg.train_acc_run_avg)
        wandb.log({'epoch': epoch, 'train_avg_loss': run_avg.train_loss_run_avg, 'train_accuracy':
                   run_avg.train_acc_run_avg})





