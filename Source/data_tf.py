__doc__ = """
Data loading and manipulation
"""

import meta
from scipy.io import loadmat
import tensorflow as tf
import numpy as np


def get_dataset(horizon=5):
    """
    Fetches the TF dataset with correct horizon, ready to be fitted into keras

    :param horizon: the prediction horizon (1 -> next 1, 2 -> next 2 , 3 -> next 3, 4 -> next 5, 5 -> next 10)
    :return: training TF dataset
    :return: test TF dataset
    :return: np array of test labels
    """
    train_x, train_y, test_x, test_y, _, true_labels = load_data(horizon)
    train_dataset, test_dataset = make_dataset(train_x, train_y, test_x, test_y)

    return train_dataset, test_dataset, true_labels


def get_dataset_numpy(horizon=5):
    """
    Fetches the dataset in form of numpy matrices with correct horizon, ready to be fitted into keras

    :param horizon: the prediction horizon (1 -> next 1, 2 -> next 2 , 3 -> next 3, 4 -> next 5, 5 -> next 10)
    :return: np arrays of training data and labels
    :return: np arrays of test data and labels
    """
    train_x, train_y, test_x, test_y, _, _ = load_data(horizon)

    return train_x, train_y, test_x, test_y


def load_data(horizon):
    """
    Load the data from .mat files.
    Data is in format: 7 days of training, 3 days of test. This is Setup 2 from TABL paper.

    :return: np array of train data
    :return: np array of train labels oneHot encoded
    :return: np array of test data
    :return: np array of test labels oneHot encoded
    :return: np array of training labels categorical
    :return: np array of test labels categorical
    """
    # X -> 254,651 training tensors
    # Y -> 5 labels for prediction horizon of (1, 2, 3, 5, 10)
    train_data = loadmat(meta.train_path)
    # X -> 139,290 test tensors
    test_data = loadmat(meta.test_path)

    # Transform the data into split numpy arrays
    train_x = train_data['x_train']
    train_x = np.float16(train_x)

    train_y = train_data['y_train'][:, horizon - 1]
    train_y_cat = tf.keras.utils.to_categorical(y=train_y - 1, num_classes=3)
    train_y_cat = np.float16(train_y_cat)

    test_x = test_data['x_test']
    test_x = np.float16(test_x)

    test_y = test_data['y_test'][:, horizon - 1]
    test_y_cat = tf.keras.utils.to_categorical(y=test_y - 1, num_classes=3)
    test_y_cat = np.float16(test_y_cat)

    return train_x, train_y_cat, test_x, test_y_cat, train_y - 1, test_y - 1


def make_dataset(train_x, train_y, test_x, test_y):
    """
    Create TF datasets from numpy arrays

    :param train_x: np array of training data
    :param train_y: np array of training labels oneHot
    :param test_x: np array of test data
    :param test_y: np array of test labels oneHot
    :return: TF training dataset
    :return: TF test dataset
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    train_dataset = train_dataset.shuffle(buffer_size=meta.pretrain_batch_size,
                                          reshuffle_each_iteration=True,
                                          seed=100)

    train_dataset.repeat(np.ceil(train_x.shape[0] / meta.tabl_batch_size))
    train_dataset = train_dataset.batch(meta.tabl_batch_size)
    test_dataset = test_dataset.batch(meta.tabl_batch_size)

    return train_dataset, test_dataset


if __name__ == '__main__':
    get_dataset()