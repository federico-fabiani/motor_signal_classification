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
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
import logging
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
                              filters_size=[int(my_params['size_1']), int(my_params['size_2']), int(my_params['size_3'])],
                              dropout=my_params['frac_dropout'],
                              units=int(my_params['n_units']),
                              neurons=my_params['n_neurons'])

    return my_network, my_params


def tune_learning_rate(model_to_tune, my_model, training_set, training_labels, validation_set, validation_labels):
    """Tuning function will try several LRs and will plot how the loss chances according to them; select as LR the last
    one before the loss explode"""
    import ktrain
    # To tune learning rate, select the last LR before loss explodes
    model_to_tune.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-3),
                          metrics=['accuracy'])
    learner = ktrain.get_learner(my_model, train_data=(training_set, training_labels),
                                 val_data=(validation_set, validation_labels), batch_size=16)
    learner.lr_find()
    learner.lr_plot()


def training_phase(my_model, training_set, training_labels, validation_set, validation_labels, K=None):
    if K is not None:
        training_set = np.concatenate((training_set, validation_set))
        training_labels = np.concatenate((training_labels, validation_labels))
        kfold = KFold(n_splits=K, shuffle=True)

    fold_no = 1
    plt.figure()
    plt.title(f'{FILE}_{EPOCH}_{outputs}_Training')
    my_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=LR), metrics=['accuracy'])
    my_model.save_weights(f'temp/{EPOCH}_{outputs}_init.hdf5')
    es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=10)

    if K is not None:
        plt.ylabel('val_loss')
        k_accuracy = []
        for train, val in kfold.split(training_set, training_labels):
            print(f'Training fold N {fold_no}:')
            my_model.reset_states()
            my_model.load_weights(f'temp/{EPOCH}_{outputs}_init.hdf5')
            checkpoint = keras.callbacks.ModelCheckpoint(f'temp/{EPOCH}_{outputs}_{fold_no}.hdf5', monitor='val_loss',
                                                         save_best_only=True)
            history = my_model.fit(training_set[train], training_labels[train],
                                   validation_data=(training_set[val], training_labels[val]),
                                   epochs=100, batch_size=16, verbose=0, callbacks=[checkpoint, es])

            # plt.plot(history.history['accuracy'], label='train')
            plt.plot(history.history['val_loss'], label=f'{fold_no}')

            print(f'Best accuracy fold {fold_no}: {max(history.history["val_accuracy"])}\n')
            fold_no = fold_no + 1
            k_accuracy.append(max(history.history["val_accuracy"]))
        print(f'average accuracy: {sum(k_accuracy) / len(k_accuracy)}')

    else:
        checkpoint = keras.callbacks.ModelCheckpoint(f'temp/{EPOCH}_{outputs}_None.hdf5', monitor='val_loss',
                                                     save_best_only=True)
        history = my_model.fit(training_set, training_labels, validation_data=(validation_set, validation_labels),
                               epochs=100, batch_size=16, verbose=1,
                               callbacks=[checkpoint, es])

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label=f'val')
        plt.ylabel('training loss')

    plt.xlabel('epoch')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'../plots/{FILE}_{EPOCH}({outputs})_K({K})_fold_training.png')
    plt.show()


def average_weights(weights_list):
    n_models = len(weights_list)
    weights = [1 / n_models for i in range(1, n_models + 1)]
    avg_model_weights = list()
    n_layers = len(weights_list[0])
    for layer in range(n_layers):
        # collect this layer from each model
        layer_weights = np.array([weights_list[layer] for i in range(n_models)])
        # weighted average of weights for this layer
        avg_layer_weights = np.average(layer_weights, axis=0, weights=weights)
        # store average layer weights
        avg_model_weights.append(avg_layer_weights)
    return avg_model_weights


def confusion_matrix(predictions, true_values, one_hot_encoder=None, plot=True):
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
        size = 8 + (outputs - 6) / 3
        fig = plt.figure(figsize=(size, size))
        ax = fig.add_subplot()
        plt.title(f'{FILE}_{EPOCH}_{outputs}_Confusion Matrix')
        im = plt.imshow(conf_matrix_norm, cmap='Reds')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        for i in range(outputs):
            for j in range(outputs):
                c = conf_matrix[i][j]
                if c > 0:
                    ax.text(j, i, str(c), va='center', ha='center', c='white')
        if not grouped_category:
            ax.vlines(seps, -0.5, outputs - 0.5, linewidth=2)
            ax.hlines(seps, -0.5, outputs - 0.5, linewidth=2)
        ax.vlines([i + 0.5 for i in range(outputs)], -0.5, outputs - 0.5, linewidth=0.5, colors='Gray')
        ax.hlines([i + 0.5 for i in range(outputs)], -0.5, outputs - 0.5, linewidth=0.5, colors='Gray')
        plt.xticks(ticks=[i for i in range(outputs)], labels=one_hot_encoder.classes_)
        plt.yticks(ticks=[i for i in range(outputs)], labels=one_hot_encoder.classes_)
        plt.ylabel('object')
        plt.xlabel('prediction')
        plt.tight_layout()
        plt.savefig(f'../plots/{FILE}_{EPOCH}_{outputs}_conf_matrix.png')
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
    K = 10  # None for no Kfold

    """ Use the object selector to select the shapes to use as classes; specify whether to group them by shape. The 
    possible shapes are: 'mixed', 'rings', 'boxes', 'balls', 'cubes', 'strength', 'precision', 'vCylinder', 'hCylinder', 
    'special', 'special2', 'strength', 'precision' """
    selector = ObjectSelector()
    # new_classes = selector.get_shapes(['rings'], group_labels=False)
    new_classes = selector.get_non_special(group_labels=False)
    # new_classes = selector.get_all(group_labels=False)
    logging.info(f'Number of classes: {len(new_classes)}\n\t{new_classes}')

    # Loading dataset
    measurements, objects, trial_states = load_dataset(PATH, EPOCH)

    # Preprocessing measurements
    label_encoder = LabelEncoder()
    X, Y = preprocess_dataset(measurements, objects, labelled_classes=new_classes, one_hot_encoder=label_encoder,
                              norm_classes=False)

    (_, channels, window) = X.shape
    outputs = Y.shape[1]

    # Splitting sets
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_sets(X, Y, tr_split=0.85, val_split=0.15,
                                                                      repetitions=K)

    # Generating model
    # (_, channels, samples) = X.shape
    # outputs = Y.shape[1]
    # model = EEGNet(channels, samples, outputs)
    # model.summary()
    # training_phase(model, x_train[0], y_train[0], x_val[0], y_val[0])
    # model.load_weights(f'temp/{EPOCH}_{outputs}_None.hdf5')
    # prediction = model.predict(x_test)
    # confusion_matrix(prediction, y_test, one_hot_encoder=label_encoder, plot=True)
    # accuracy = accuracy_score(y_true=y_test.argmax(axis=1), y_pred=prediction.argmax(axis=1))
    # print(accuracy)

    networks_to_try = ['EEGNetv2']  # ['DenseNN', 'SimpleRNN', 'GRU', 'LSTM', 'CNN', 'EEGNetv2']
    with open('../utils/hyperparameters.json', 'r') as f:
        config = json.load(f)

    for net_name in networks_to_try:
        print(f'### {net_name} ###')
        network, params = get_decoder(window, channels, outputs, net_name, config)
        network.model.summary()

        net_results = []
        print('Repetition: ', end='\t')
        for k in range(K):
            network.reset_weights()
            network.fit(x_train[k], y_train[k], num_epochs=int(params['n_epochs']),
                        batch_size=int(params['batch_size']))
            prediction = network.predict(x_val[k])
            net_results.append(accuracy_score(y_true=y_val[k].argmax(axis=1), y_pred=prediction.argmax(axis=1)))
            print(f'{k + 1}/{K} [{round(net_results[-1], 3)}]', end='\t')
        net_results = np.array(net_results)
        print(f'\nAccuracy [mean | std] : {net_results.mean()} | {net_results.std()}\n')
