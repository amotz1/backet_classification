import torch
from torch import nn


def evaluation(args, model, valid_generator):
    model.eval()
    device = args.device
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    valid_loss, correct,total = 0,0,0

    with torch.no_grad():
        for batch_index, input_tensor in enumerate(valid_generator):
            input_data, target = input_tensor
            input_data, target = input_data.to(device), target.to(device)
            output = model(input_data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            valid_loss += criterion(output, target).item()

            print('Test Accuracy of the model on the 10000 test images: {} %'
                  .format(100 * correct / total), f'average loss is {valid_loss/total}')














