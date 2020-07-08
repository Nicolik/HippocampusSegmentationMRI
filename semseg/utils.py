import os
import numpy as np

from config import train_images_folder, train_labels_folder


def dice_coeff(gt, pred, eps=1e-5):
    dice = np.sum(pred[gt == 1]) * 2.0 / (np.sum(pred) + np.sum(gt))
    return dice


def multi_dice_coeff(gt, pred, num_classes):
    labels = one_hot_encode_np(gt, num_classes)
    outputs = one_hot_encode_np(pred, num_classes)
    dices = list()
    for cls in range(1, num_classes):
        outputs_ = outputs[:, cls]
        labels_  = labels[:, cls]
        dice_ = dice_coeff(outputs_, labels_)
        dices.append(dice_)
    return sum(dices) / (num_classes-1)


def one_hot_encode_np(label, num_classes):
    """ Numpy One Hot Encode
    :param label: Numpy Array of shape BxHxW or BxDxHxW
    :param num_classes: K classes
    :return: label_ohe, Numpy Array of shape BxKxHxW or BxKxDxHxW
    """
    assert len(label.shape) == 3 or len(label.shape) == 4, 'Invalid Label Shape {}'.format(label.shape)
    label_ohe = None
    if len(label.shape) == 3:
        label_ohe = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2]))
    elif len(label.shape) == 4:
        label_ohe = np.zeros((label.shape[0], num_classes, label.shape[1], label.shape[2], label.shape[3]))
    for batch_idx, batch_el_label in enumerate(label):
        for cls in range(num_classes):
            label_ohe[batch_idx, cls] = (batch_el_label == cls)
    return label_ohe


def train_val_split(train_images, train_labels, train_index, val_index, do_join=True):
    train_images_np, train_labels_np = np.array(train_images), np.array(train_labels)
    train_images_list = list(train_images_np[train_index])
    val_images_list = list(train_images_np[val_index])
    train_labels_list = list(train_labels_np[train_index])
    val_labels_list = list(train_labels_np[val_index])
    if do_join:
        train_images_list = [os.path.join(train_images_folder, train_image) for train_image in train_images_list]
        val_images_list = [os.path.join(train_images_folder, val_image) for val_image in val_images_list]
        train_labels_list = [os.path.join(train_labels_folder, train_label) for train_label in train_labels_list]
        val_labels_list = [os.path.join(train_labels_folder, val_label) for val_label in val_labels_list]
    return train_images_list, val_images_list, train_labels_list, val_labels_list


def plot_confusion_matrix(cm,
                          target_names=None,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          already_normalized=False,
                          path_out=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))
    plt.matshow(cm, cmap=cmap)
    plt.title(title, pad=25.)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize or already_normalized else cm.max() / 2
    print("Thresh = {}".format(thresh))
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize or already_normalized:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if path_out is not None:
        plt.savefig(path_out)
    plt.show()
