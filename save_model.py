import os
import torch


def save_model(model, optimizer, args, epoch):
    save_path = args.save
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    torch.save({'epoch': epoch, 'mode_state_dict': model.state_dict,
                'optimizer_state_dict': optimizer.state_dict}, save_path + '/' + 'backet_net.pt')

