import torch
from torch import nn
import time

def train(args, model, train_loader, epoch, optimizer):
    model.train()
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss(reduction='mean')

    for batch_index, input_tensor in enumerate(train_loader):
        input_data, target = input_tensor
        print('target.shape ', target.shape)
        print('len target', len(target))
        print('target[0]', target[0])
        print('target[1]', target[1])
        print('target[2]', target[2])
        print('target[3]', target[3])
        time.sleep(3)

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
        
        print('output.shape', output.shape)
        print('pred[0] = ', pred[0])
        print('pred[1] = ', pred[1])
        print('pred[2] = ', pred[2])
        print('pred[3] = ', pred[3])
        time.sleep(3)

        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
        return correct, correct/len(target)



