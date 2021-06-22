import torch
from torch import nn
import wandb
from utils import RunningAverage
from utils import acc

def evaluation(args, model, valid_loader, epoch, scaler, run_avg):
    model.eval()
    device = args.device
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    with torch.no_grad():
        for batch_index, input_tensor in enumerate(valid_loader):
            input_data, target = input_tensor
            input_data, target = input_data.to(device), target.to(device)

            with torch.cuda.amp.autocast():
                output = model(input_data)
                valid_loss = criterion(output, target)

            run_avg = RunningAverage()
            run_avg.update_train_loss_avg(valid_loss.item(), args.batch_size)
            _, predicted = torch.max(output.data, 1)
            accuracy = acc(output, target)
            run_avg.update_train_acc_avg(accuracy, args.batch_size)
            wandb.log({'epoch': epoch, 'valid_avg_loss': run_avg.val_loss_run_avg,
                       'val_accuracy': run_avg.val_acc_run_avg})













