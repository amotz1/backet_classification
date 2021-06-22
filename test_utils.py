
from utils import RunningAverage


def test_running_average():
    train_losses = [1, 0.5, 0.3]
    train_accuracies = [0.3, 0.5, 0.1]
    running_average = RunningAverage()
    for i in range(len(train_losses)):
        running_average.update_train_loss_avg(train_losses[i], 1)
        running_average.update_train_acc_avg(train_accuracies[i], 1)

    assert running_average.train_loss_run_avg == 0.6
    assert running_average.train_acc_run_avg == 0.3

    running_average.update_train_loss_avg(0.2, 1)
    running_average.update_train_acc_avg(0.1, 1)

    assert running_average.train_loss_run_avg == 0.5
    assert running_average.train_acc_run_avg == 0.25

    val_losses = [1, 0.7, 1.3]
    val_accuracies = [0.3, 0.4, 8.3]

    for i in range(len(val_losses)):
        running_average.update_val_loss_avg(val_losses[i], 1)
        running_average.update_val_acc_avg(val_accuracies[i], 1)

    assert running_average.val_loss_run_avg == 1
    assert running_average.val_acc_run_avg == 3

    running_average.update_val_loss_avg(3, 1)
    running_average.update_val_acc_avg(7, 1)

    assert running_average.val_loss_run_avg == 1.5
    assert running_average.val_acc_run_avg == 4

    running_average.reset_train()
    running_average.reset_val()

    assert running_average.sum_train_loss == 0
    assert running_average.sum_train_acc == 0
    assert running_average.train_loss_counter == 0
    assert running_average.train_acc_counter == 0
    assert running_average.train_loss_run_avg == 0
    assert running_average.train_acc_run_avg == 0

    assert running_average.sum_val_loss == 0
    assert running_average.sum_val_acc == 0
    assert running_average.val_loss_counter == 0
    assert running_average.val_acc_counter == 0
    assert running_average.val_loss_run_avg == 0
    assert running_average.val_acc_run_avg == 0


test_running_average()