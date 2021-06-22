# tensorflow-gpu version 2.4.1
# CUDA version 11.0
# Cudnn version 8.0.5
# Python 3.8

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow import keras, get_logger

get_logger().setLevel('ERROR')

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from utils.data_processing import *
from utils.functions import *
from utils.decoders import *

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json


def get_decoder(n_samples, n_channels, n_outputs, decoder_name, config_dict):
    (my_network, my_params) = (None, None)

    if decoder_name == 'DenseNN':
        my_params = config_dict['dnn']
        my_network = DenseNNClassification(n_samples, n_channels, n_outputs,
                                           units=[my_params['num_units_1'], my_params['num_units_2']],
                                           dropout=my_params['frac_dropout'])

    if decoder_name == 'SimpleRNN':
        my_params = config_dict['rnn']
        my_network = SimpleRNNClassification(n_samples, n_channels, n_outputs,
                                             units=my_params['num_units'],
                                             dropout=my_params['frac_dropout'])

    if decoder_name == 'GRU':
        my_params = config_dict['gru']
        my_network = GRUClassification(n_samples, n_channels, n_outputs,
                                       units=my_params['num_units'],
                                       dropout=my_params['frac_dropout'])

    if decoder_name == 'LSTM':
        my_params = config_dict['lstm']
        my_network = LSTMClassification(n_samples, n_channels, n_outputs,
                                        units=my_params['num_units'],
                                        dropout=my_params['frac_dropout'])

    if decoder_name == 'CNN':
        my_params = config_dict['cnn']
        my_network = CNNClassification(n_samples, n_channels, n_outputs,
                                       filters=my_params['num_filters'],
                                       size=my_params['kernel_size'],
                                       dropout=my_params['frac_dropout'],
                                       pool_size=2)

    if decoder_name == 'EEGNet':
        # my_params = config_dict['eeg_net']
        my_params = {'fac_dropout': 0.1, 'n_epochs': 10, 'batch_size': 6}
        my_network = EEGNet(dropout=my_params['fac_dropout'])

    if decoder_name == 'EEGNetv2':
        my_params = config_dict['eegnet2']
        my_network = EEGNetv2(n_channels, n_outputs,
                              filters=[my_params['n_filters_1'], my_params['n_filters_2'], my_params['n_filters_3']],
                              filters_size=[int(my_params['size_1']), int(my_params['size_2']),
                                            int(my_params['size_3'])],
                              dropout=my_params['frac_dropout'],
                              units=int(my_params['n_units']),
                              neurons=my_params['n_neurons'])

    return my_network, my_params


def confusion_matrix(true_values, predictions, my_outputs, one_hot_encoder=None, plot=True):
    """Horrible function, but it does its task"""
    # i (rows) are true values and j (columns) are the prediction, therefore you can select on the row the object of
    # interest and check on the columns how they were classified
    if one_hot_encoder is None:
        labels = set(np.concatenate(predictions, true_values))
    else:
        labels = list(one_hot_encoder.classes_)
        predictions = one_hot_encoder.inverse_transform(predictions.argmax(axis=1))
        true_values = one_hot_encoder.inverse_transform(true_values.argmax(axis=1))

    conf_matrix = np.zeros(shape=(len(labels), len(labels)))
    for h in range(len(true_values)):
        row = labels.index(true_values[h])
        col = labels.index(predictions[h])
        conf_matrix[row][col] += 1

    if plot:
        conf_matrix_norm = normalize(conf_matrix, axis=1, norm='l1')
        grouped_category = False if labels[0].isdigit() else True
        if not grouped_category:
            seps = []
            past_label = labels[0][0]
            for i, l in enumerate(labels):
                if len(l) == 1 or l[0] != past_label:
                    seps.append(int(i) - 0.5)
                past_label = l[0]
        size = 8 + (my_outputs - 6) / 3
        fig = plt.figure(figsize=(size, size))
        ax = fig.add_subplot()
        plt.title(f'{FILE}_{EPOCH}_{my_outputs}_Confusion Matrix')
        im = plt.imshow(conf_matrix_norm, cmap='Reds')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        for i in range(my_outputs):
            for j in range(my_outputs):
                c = conf_matrix[i][j]
                if c > 0:
                    ax.text(j, i, str(c), va='center', ha='center', c='white')
        if not grouped_category:
            ax.vlines(seps, -0.5, my_outputs - 0.5, linewidth=2)
            ax.hlines(seps, -0.5, my_outputs - 0.5, linewidth=2)
        ax.vlines([i + 0.5 for i in range(my_outputs)], -0.5, my_outputs - 0.5, linewidth=0.5, colors='Gray')
        ax.hlines([i + 0.5 for i in range(my_outputs)], -0.5, my_outputs - 0.5, linewidth=0.5, colors='Gray')
        plt.xticks(ticks=[i for i in range(my_outputs)], labels=one_hot_encoder.classes_)
        plt.yticks(ticks=[i for i in range(my_outputs)], labels=one_hot_encoder.classes_)
        plt.ylabel('object')
        plt.xlabel('prediction')
        plt.tight_layout()
        plt.savefig(f'../plots/{FILE}_{EPOCH}_{my_outputs}_conf_matrix.png')
        plt.show()
    return conf_matrix


if __name__ == '__main__':
    np.random.seed(27)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        format='%(message)s',
        filename=f'../log/{datetime.now().strftime("%Y_%m_%d_%H_%M.log")}'
    )
    clean_logs('../log', keep=5)

    """Classification setting: Select the file from which takes measurements, the epoch of interest, and decide which 
    object could be considered as a class: classes is a list, in which each entry is one or more object composing a 
    class (this is still not really clean nor elegant, but the idea is that of grouping objects by shape and reduce 
    the number of classes) """

    FILE = 'ZRec50_Mini'  # MRec40, ZRec50 or ZRec50_Mini
    PATH = f'../data/Objects Task DL Project/{FILE}.neo.mat'
    EPOCH = 'all'  # ['start', 'rest', 'motor', 'fixlon', 'fix', 'cue', 'mem', 'react', 'go', 'hold', 'rew',
    # 'intert', 'end']
    LR = 7e-3  # None for tuning lr
    K = 1  # None for no Kfold

    """ Use the object selector to select the shapes to use as classes; specify whether to group them by shape. The 
    possible shapes are: 'mixed', 'rings', 'boxes', 'balls', 'cubes', 'strength', 'precision', 'vCylinder', 'hCylinder', 
    'special', 'special2', 'strength', 'precision' """
    selector = ObjectSelector()
    new_classes = selector.get_shapes(['precision', 'strength'], group_labels=False)
    # new_classes = selector.get_non_special(group_labels=False)
    # new_classes = selector.get_all(group_labels=False)
    # new_classes = [(['23', '24', '25'], 'rings'), (['43', '44', '45'], 'balls'), (['63', '64', '65'], 'boxes')]
    logging.info(f'Number of classes: {len(new_classes)}\n\t{new_classes}')

    # Loading dataset
    measurements, objects, trial_states = load_dataset(PATH, EPOCH)

    # Preprocessing measurements
    object_encoder = LabelEncoder()
    X, Y = preprocess_dataset(measurements, objects, labelled_classes=new_classes, one_hot_encoder=object_encoder,
                              norm_classes=True)
    state_encoder = LabelEncoder()
    X, Y, STATES = expanding_window_preprocessing(X, Y, trial_states, state_encoder, norm_classes=True)

    (n_trials, channels, window) = X.shape
    bit_objects = Y.shape[1]
    bit_states = STATES.shape[1]

    Y_STATES = np.array([np.hstack((Y[i, :], STATES[i, :])) for i in range(n_trials)])

    OUT = Y_STATES  # or Y or STATES
    outputs = OUT.shape[1]

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_sets(X, OUT, tr_split=0.7, val_split=0.3,
                                                                      repetitions=K)
    print(x_train[0].shape, y_train[0].shape, x_val[0].shape, y_val[0].shape)

    networks_to_try = ['EEGNetv2']  # ['DenseNN', 'SimpleRNN', 'GRU', 'LSTM', 'CNN', 'EEGNetv2']
    with open('../utils/hyperparameters.json', 'r') as f:
        config = json.load(f)

    for net_name in networks_to_try:
        print(f'### {net_name} ###')
        network, params = get_decoder(window, channels, outputs, net_name, config)
        network.model.summary()
        network.reset_weights()
        history = network.fit(x_train[0], y_train[0], x_val[0], y_val[0], num_epochs=1000,
                              batch_size=int(params['batch_size']), verbose=1)
        prediction = network.predict(x_val[0])
        confusion_matrix(y_val[0][:, :bit_objects], prediction[:, :bit_objects], bit_objects, one_hot_encoder=object_encoder, plot=True)
        confusion_matrix(y_val[0][:, bit_objects:], prediction[:, bit_objects:], bit_states, one_hot_encoder=state_encoder, plot=True)
        accuracy_obj = accuracy_score(y_true=y_val[0][:, :bit_objects].argmax(axis=1), y_pred=prediction[:, :bit_objects].argmax(axis=1))
        accuracy_state = accuracy_score(y_true=y_val[0][:, bit_objects:].argmax(axis=1), y_pred=prediction[:, bit_objects:].argmax(axis=1))
        print(f'Accuracy score: obj {accuracy_obj} -- state {accuracy_state}')

