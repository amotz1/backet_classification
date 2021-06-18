import torch
from torch import nn
import time
import wandb


def train(args, model, train_loader, epoch, optimizer, scaler):
    model.train()
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss(reduction='mean')
    total_loss = 0

    for batch_index, input_tensor in enumerate(train_loader):
        input_data, target = input_tensor

        if args.cuda:
            input_data = input_data.to(device)
            target = target.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(input_data)
            loss = criterion(output, target)
        total_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        correct, accuracy = acc(output, target)
        wandb.log({'epoch': epoch, 'train_avg_loss': total_loss/(batch_index+1), 'train_accuracy':
                   correct/(batch_index+1)})


def acc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)

        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
        return correct, correct/len(target)



