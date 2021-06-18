import torch
from torch import nn
import wandb


def evaluation(args, model, valid_loader, epoch, scaler):
    model.eval()
    device = args.device
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    valid_loss, correct, num_samples = 0,0,0

    with torch.no_grad():
        for batch_index, input_tensor in enumerate(valid_loader):
            num_samples += args.batch_size
            input_data, target = input_tensor
            input_data, target = input_data.to(device), target.to(device)

            with torch.cuda.amp.autocast():
                output = model(input_data)
                valid_loss += criterion(output, target).item()

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            wandb.log({'epoch':epoch, 'valid_avg_loss': valid_loss/num_samples,
                       'val_accuracy': correct/num_samples})













