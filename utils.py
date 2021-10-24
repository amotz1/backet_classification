import torch
import os
import argparse


class RunningAverage:
    def __init__(self):
        self.sum_train_loss = 0
        self.sum_train_acc = 0
        self.train_loss_counter = 0
        self.train_acc_counter = 0
        self.train_loss_run_avg = 0
        self.train_acc_run_avg = 0

        self.sum_val_loss = 0
        self.sum_val_acc = 0
        self.val_loss_counter = 0
        self.val_acc_counter = 0
        self.val_loss_run_avg = 0
        self.val_acc_run_avg = 0

    def update_train_loss_avg(self, train_loss, batch_size):
        self.sum_train_loss += train_loss*batch_size
        self.train_loss_counter += batch_size
        self.train_loss_run_avg = self.sum_train_loss / self.train_loss_counter

    def update_train_acc_avg(self, train_acc, batch_size):
        self.sum_train_acc += train_acc*batch_size
        self.train_acc_counter += batch_size
        self.train_acc_run_avg = self.sum_train_acc / self.train_acc_counter

    def update_val_loss_avg(self, val_loss, batch_size):
        self.sum_val_loss += val_loss*batch_size
        self.val_loss_counter += batch_size
        self.val_loss_run_avg = self.sum_val_loss / self.val_loss_counter

    def update_val_acc_avg(self, val_acc, batch_size):
        self.sum_val_acc += val_acc*batch_size
        self.val_acc_counter += batch_size
        self.val_acc_run_avg = self.sum_val_acc / self.val_acc_counter

    def reset_train(self):
        self.sum_train_loss = 0
        self.sum_train_acc = 0
        self.train_loss_counter = 0
        self.train_acc_counter = 0
        self.train_loss_run_avg = 0
        self.train_acc_run_avg = 0

    def reset_val(self):
        self.sum_val_loss = 0
        self.sum_val_acc = 0
        self.val_loss_counter = 0
        self.val_acc_counter = 0
        self.val_loss_run_avg = 0
        self.val_acc_run_avg = 0


def acc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)

        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
        return correct/len(target)


def save_checkpoint(model, optimizer, args, epoch):
    save_path = args.save
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, save_path + '/' + 'backet_net.pt')


def load_checkpoint(check_point):
    checkpoint = torch.load(check_point)

    m_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']

    return m_state_dict, optimizer_state_dict


def get_args():
    print("hi")
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--dataset_name', type=str, default='shape_class', help='dataset_name')
    parser.add_argument('--epochs_number', type=int, default=8, help='epoch number for training')
    parser.add_argument('--classes', type=int, default=10, help='dataset_classes')
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
    parser.add_argument('--device', type=int, default=0, help='gpu device')  # default is 0
    parser.add_argument('--weight_decay', type=int, default=1e-7, help='dropout')
    parser.add_argument('--cuda', action='store_true', default=True, help='gpu for training acceleration')
    parser.add_argument('--model', type=str, default='FullyConnected')
    parser.add_argument('--opt', type=str, default='adamw', choices=('sgd','adam','rmsprop'))
    parser.add_argument('--root_path', type=str, default='/content/backet_classification/MnistData', help='path to dataset')
    parser.add_argument('--save', type=str, default='/content/save/backetnet', help='path to checkpoint save directory')

    args = parser.parse_args()
    return args

