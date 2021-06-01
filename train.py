import torch
from torch import nn


def train(args, model, train_generator, epoch, optimizer):
    model.train()
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss(reduction='mean')

    for batch_index, input_tensor in enumerate(train_generator):
        input_data, target = input_tensor

        if args.cuda:
            input_data = input_data.to(device)
            target = target.to(device)

        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        correct, accuracy = acc(output, target)
        if batch_index % 100 == 0:
            print(f'epoch: {epoch} loss:{loss.item()} acc:{accuracy}')



def acc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
        return correct, correct/len(target)



