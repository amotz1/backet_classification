import torch
from torch import nn


def evaluation(args, model, valid_generator):
    model.eval()
    device = args.device
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    valid_loss, correct = 0,0

    with torch.no_grad():
        for batch_index, input_tensor in enumerate(valid_generator):
            input_data, target = input_tensor
            input_data, target = input_data.to(device), target.to(device)

            output = model(input_data)
            valid_loss += criterion(output, target).item()
            correct += (output.argmax(1) == target).type(torch.float.sum().item())











