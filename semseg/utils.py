import torch
import os
import time
import numpy as np
from semseg.loss import get_multi_dice_loss, LEARNING_RATE_REDUCTION_FACTOR, one_hot_encode

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


def dice_coeff(gt, pred, eps=1e-5):
    dice = np.sum(pred[gt == 1]) * 2.0 / (np.sum(pred) + np.sum(gt))
    return dice


def multi_dice_coeff(gt, pred, num_classes):
    labels = one_hot_encode_np(gt, num_classes)
    outputs = one_hot_encode_np(pred, num_classes)
    dices = list()
    for cls in range(1, num_classes):
        outputs_ = outputs[cls]
        labels_  = labels[cls]
        dice_ = dice_coeff(outputs_, labels_)
        dices.append(dice_)
    return sum(dices) / (num_classes-1)


def one_hot_encode_np(label, num_classes):
    """

    :param label: Numpy Array of shape HxW or DxHxW
    :param num_classes: K classes
    :return: label_ohe, Numpy Array of shape KxHxW or KxDxHxW
    """
    assert len(label.shape) == 2 or len(label.shape) == 3, 'Invalid Label Shape {}'.format(label.shape)
    if len(label.shape) == 2:
        label_ohe = np.zeros((num_classes, label.shape[0], label.shape[1]))
    elif len(label.shape) == 3:
        label_ohe = np.zeros(( num_classes, label.shape[0], label.shape[1], label.shape[2]))
    for cls in range(num_classes):
        label_ohe[cls] = (label == cls)
    return label_ohe