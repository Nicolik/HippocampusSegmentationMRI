import torch
import os
import time
import numpy as mp
from semseg.data_loader import SemSegConfig, Dataset3DFull
from semseg.loss import get_multi_dice_loss, LEARNING_RATE_REDUCTION_FACTOR

def GetDataLoader3D(config: SemSegConfig) -> torch.utils.data.DataLoader:
    print('Building Training Set Loader...')
    train = Dataset3DFull(config.train_images, config.train_labels,
                          augmentation=config.augmentation, do_normalize=config.do_normalize,
                          zero_pad=config.zero_pad, pad_ref=config.pad_ref)

    train_data = torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True,
                                             num_workers=config.num_workers)
    print('Training Loader built!')
    return train_data


def min_max_normalization(input):
    return (input - input.min()) / (input.max() - input.min())


def train_model(net, optimizer, train_data, config,
                device=None, logs_folder=None):

    print('Start training...')
    # train loop
    for epoch in range(config.epochs):

        epoch_start_time = time.time()
        running_loss = 0.0

        # lower learning rate
        if epoch == config.low_lr_epoch:
            for param_group in optimizer.param_groups:
                config.lr = config.lr / LEARNING_RATE_REDUCTION_FACTOR
                param_group['lr'] = config.lr

        # switch to train mode
        net.train()

        for i, data in enumerate(train_data):

            # wrap data in Variables
            inputs, labels = data
            if config.cuda: inputs, labels = inputs.cuda(), labels.cuda()

            # forward pass and loss calculation
            outputs = net(inputs)

            # get multi dice loss
            loss = get_multi_dice_loss(outputs, labels, device=device)

            # empty gradients, perform backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save and print statistics
            running_loss += loss.data

        epoch_end_time = time.time()
        epoch_elapsed_time = epoch_end_time - epoch_start_time

        # print statistics
        print('  [Epoch {:04d}] - Train dice loss: {:.4f} - Time: {:.1f}'
              .format(epoch + 1, running_loss / (i + 1), epoch_elapsed_time))


        # switch to eval mode
        net.eval()

        # only validate every 'val_epochs' epochs
        if epoch % config.val_epochs == 0:
            if logs_folder is not None:
                checkpoint_path = os.path.join(logs_folder, 'model_epoch_{:04d}.pht'.format(epoch))
                torch.save(net.state_dict(), checkpoint_path)

    print('Training ended!')
    return net