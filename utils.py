import torch
import os


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

    torch.save({'epoch': epoch, 'mode_state_dict': model.state_dict,
                'optimizer_state_dict': optimizer.state_dict}, save_path + '/' + 'backet_net.pt')


def load_checkpoint(check_point):
    torch.load(check_point)

    model = model.load_state_dict(check_point['state_dict'])
    print(model)
    optimizer = optimizer.load_state_dict(check_point['optimizer'])

    return model, optimizer