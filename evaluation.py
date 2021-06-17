import torch
from torch import nn
import wandb

def evaluation(args, model, valid_loader, train_avg_loss, train_accuracy, epoch):
    model.eval()
    device = args.device
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    valid_loss, correct,total = 0,0,0

    with torch.no_grad():
        for batch_index, input_tensor in enumerate(valid_loader):
            input_data, target = input_tensor
            input_data, target = input_data.to(device), target.to(device)
            output = model(input_data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            valid_loss += criterion(output, target).item()
            wandb.log({'epoch':epoch,'valid_avg_loss': valid_loss/args.batch_size, 'accuracy': correct/total})













