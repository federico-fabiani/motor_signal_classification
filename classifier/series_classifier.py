# tensorflow-gpu version 2.4.1
# CUDA version 11.0
# Cudnn version 8.0.5
# Python 3.8

from tensorflow import keras
import numpy as np
from utils.data_processing import *
from utils.object_selector import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
import logging
from datetime import datetime
import os


def clean_logs(log_path, keep=5):
    logs = os.listdir(log_path)
    while len(logs) > keep:
        os.remove(log_path + '/' + logs[0])
        logs.pop(0)


def split_sets(my_x, my_y, tr_split, val_split, shuffle=True, normalize_sets=True, one_hot_encoder=None, labelled_classes=None,
               normalize_classes=True):
    # x must be a numpy array where the first dimension is the number of inputs

    if labelled_classes is not None:
        # re-associate labels according to the desired rule, recall that each entry of labels array is the list of
        # objects composing that class
        new_y = []
        new_x = []
        for y_idx, old_y in enumerate(my_y):
            for (new_class_elements, new_class_label) in labelled_classes:
                if type(new_class_elements) != list:
                    new_class_elements = [new_class_elements]
                if str(old_y) in new_class_elements:
                    new_y.append(new_class_label)
                    new_x.append(my_x[y_idx])
                    break
        my_y = np.array(new_y)
        logging.info(f'{len(new_x)}/{len(my_x)} recordings kept belonging to {len(labelled_classes)} classes')
        my_x = np.array(new_x)

    if normalize_sets:
        logging.info('Dataset normalized')
        my_x = (my_x - my_x.min()) / (my_x.max() - my_x.min())
    if shuffle:
        logging.info('Dataset shuffled')
        rnd = np.random.permutation(len(my_y))
        my_x = my_x[rnd]
        my_y = my_y[rnd]
    if normalize_classes:
        # count how many times each class is present in the dataset
        unique_y, n_repetition = np.unique(my_y, return_counts=True, axis=0)
        logging.info(f'De-biasing dataset: (unique_label, n_repetitions)\n\t{list(zip(unique_y, n_repetition))}')
        unique_y = unique_y.tolist()
        keep = min(n_repetition)
        for idx, old_y in enumerate(my_y):
            if n_repetition[unique_y.index(old_y)] > keep:
                my_y = np.delete(my_y, idx, 0)
                my_x = np.delete(my_x, idx, 0)
    if one_hot_encoder is not None:
        logging.info('Dataset encoded with one-hot labels')
        my_y = keras.utils.to_categorical(one_hot_encoder.fit_transform(my_y))

    # Checking for duplicates
    u, c = np.unique(my_x, return_counts=True, axis=0)
    dup = u[c > 1]
    if dup.shape[0] != 0:
        print('WARNING: duplicates found!')

    tr_idx = round(len(my_y) * tr_split)
    val_idx = round(len(my_y) * (tr_split + val_split))
    return (my_x[:tr_idx], my_y[:tr_idx]), \
           (my_x[tr_idx:val_idx], my_y[tr_idx:val_idx]), \
           (my_x[val_idx:], my_y[val_idx:]),


def EEGNet(features, window, n_outputs, lstm=False):
    eeg_net = keras.models.Sequential()
    eeg_net.add(keras.layers.Permute((2, 1), input_shape=(features, window)))
    eeg_net.add(keras.layers.Reshape((window, features, 1)))
    eeg_net.add(keras.layers.Conv2D(40, (15, 1), padding='same', activation=keras.activations.elu))
    eeg_net.add(keras.layers.Conv2D(40, (1, round(features / 6)), activation=keras.activations.elu))
    eeg_net.add(keras.layers.AvgPool2D((10, 1)))
    if lstm:
        eeg_net.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
        eeg_net.add(keras.layers.LSTM(32, activation='tanh', return_sequences=True))
    eeg_net.add(keras.layers.Flatten())
    eeg_net.add(keras.layers.Dense(n_outputs, activation='softmax'))
    return eeg_net


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


def tune_learning_rate(model_to_tune):
    import ktrain
    # To tune learning rate, select the last LR before loss explodes
    model_to_tune.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-3),
                          metrics=['accuracy'])
    learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_val, y_val), batch_size=16)
    learner.lr_find()
    learner.lr_plot()


def training_phase():
    pass


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
                    seps.append(int(i)-0.5)
                past_label = l[0]
        size = 8 + (outputs-6)/3
        fig = plt.figure(figsize=(size, size))
        ax = fig.add_subplot()
        plt.title(f'{FILE}_{EPOCH}_{outputs}_Confusion Matrix')
        im = plt.imshow(conf_matrix_norm, cmap='Reds')
        plt.colorbar(im,fraction=0.046, pad=0.04)
        for i in range(outputs):
            for j in range(outputs):
                c = conf_matrix[i][j]
                if c > 0:
                    ax.text(j, i, str(c), va='center', ha='center', c='white')
        if not grouped_category:
            ax.vlines(seps, -0.5, outputs-0.5, linewidth=2)
            ax.hlines(seps, -0.5, outputs-0.5, linewidth=2)
        ax.vlines([i+0.5 for i in range(outputs)], -0.5, outputs - 0.5, linewidth=0.5, colors='Gray')
        ax.hlines([i+0.5 for i in range(outputs)], -0.5, outputs - 0.5, linewidth=0.5, colors='Gray')
        plt.xticks(ticks=[i for i in range(outputs)], labels=label_encoder.classes_)
        plt.yticks(ticks=[i for i in range(outputs)], labels=label_encoder.classes_)
        plt.ylabel('object')
        plt.xlabel('prediction')
        plt.tight_layout()
        plt.savefig(f'../plots/{FILE}_{EPOCH}_{outputs}_conf_matrix.png')
        plt.show()
    return conf_matrix


if __name__ == '__main__':
    np.random.seed(1805)
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

    FILE = 'MRec40'  # MRec40, ZRec50 or ZRec50_Mini
    PATH = f'../data/Objects Task DL Project/{FILE}.neo.mat'
    EPOCH = 'hold'  # ['start', 'rest', 'motor', 'fixlon', 'fix', 'cue', 'mem', 'react', 'go', 'hold', 'rew',
    # 'intert', 'end']
    LR = 7e-3  # None
    K = 10


    """ Use the object selector to select the shapes to use as classes; specify whether to group them by shape. The 
    possible shapes are: 'mixed', 'rings', 'boxes', 'balls', 'cubes', 'strength', 'precision', 'vCylinder', 'hCylinder', 
    'special', 'special2', 'strength', 'precision' """
    selector = ObjectSelector()
    # new_classes = selector.get_shapes(['balls', 'rings', 'boxes'], group_labels=False)
    new_classes = selector.get_non_special(group_labels=False)
    # new_classes = selector.get_all(group_labels=True)
    logging.info(f'Number of classes: {len(new_classes)}\n\t{new_classes}')


    """ Load the desired measurements, from file or from cache if available """
    logging.info(f'Loading dataset at {PATH}\nSelecting epoch {EPOCH}')
    try:
        X = np.load(f'temp/{FILE}_{EPOCH}_X.npy')
        Y = np.load(f'temp/{FILE}_{EPOCH}_Y.npy')
        logging.info(f'Windows and objects loaded from cache;\n\tX - {X.shape}\n\tY - {Y.shape}')

    except IOError:
        wrapper = DataWrapper()
        wrapper.load(PATH)
        X, Y, _ = wrapper.get_epochs(EPOCH, nbins=25)
        np.save(f'temp/{FILE}_{EPOCH}_X.npy', X)
        np.save(f'temp/{FILE}_{EPOCH}_Y.npy', Y)
        logging.info('Windows and objects loaded from records;\n')

    logging.info(f'Loaded {len(Y)} records')
    # objects = list(enumerate(set(Y)))
    # print(f'{len(objects)} CLASSES:\n\t{objects}')


    """ Split the windows and objects lists into train, validation and test set """
    label_encoder = LabelEncoder()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_sets(X, Y, tr_split=0.7, val_split=0.15,
                                                                      labelled_classes=new_classes,
                                                                      normalize_sets=True,
                                                                      normalize_classes=True,
                                                                      one_hot_encoder=label_encoder)
    print('Train: ', x_train.shape, y_train.shape)
    print('Validation: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    (_, channels, samples) = x_train.shape
    outputs = y_train.shape[1]
    # t = label_encoder.inverse_transform(np.argmax(y_train, axis=1))
    model = EEGNet(channels, samples, outputs)
    model.summary()

    """The following code is necessary to print the model summary on a txt file"""
    # with open('report.txt', 'w') as fh:
    #     model.summary(print_fn=lambda x: fh.write(x + '\n'))

    """Tuning function will try several LRs and will plot how the loss chances according to them; select as LR the last 
    one before the loss explode"""
    # tune_learning_rate(model)

    # kfold = KFold(n_splits=K, shuffle=True)
    # fold_no = 1
    # k_weights = []

    plt.figure()
    plt.title(f'{FILE}_{EPOCH}_{outputs}_Training')

    # for train, val in kfold.split(x_train, y_train):
    #    print(f'Training fold N {fold_no}:')
    #    model.reset_states()
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=LR), metrics=['accuracy'])

    es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=5)
    checkpoint = keras.callbacks.ModelCheckpoint(f'temp/best_model.hdf5', monitor='val_loss',
                                                 save_best_only=True)

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=100, batch_size=16, verbose=1,
                        callbacks=[checkpoint, es])

    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label=f'val')

    # print(f'Best accuracy: {max(history.history["val_accuracy"])}\n')
    # fold_no = fold_no + 1
    # k_weights.append(model.get_weights())

    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'../plots/{FILE}_{EPOCH}_{outputs}_Training.png')
    plt.show()

    # TODO: evaluate confusion matrix and flexible way to accept as accuracy also close objects

    model.load_weights(f'temp/best_model.hdf5')

    # model.set_weights(average_weights(k_weights))
    # model.save_weights('temp/average_weights.hdf5')
    prediction = model.predict(x_test)
    confusion_matrix(prediction, y_test, one_hot_encoder=label_encoder, plot=True)
    accuracy = accuracy_score(y_true=y_test.argmax(axis=1), y_pred=prediction.argmax(axis=1))
    print(accuracy)

