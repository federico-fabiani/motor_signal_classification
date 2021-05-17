# tensorflow-gpu version 2.4.1
# CUDA version 11.0
# Cudnn version 8.0.5
# Python 3.8

from tensorflow import keras
import numpy as np
from utils.data_processing import *
from utils.object_selector import *
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
import logging
from datetime import datetime


def split_sets(my_x, my_y, tr_split, val_split, shuffle=True, normalize=True, sparse_labels=True, labels=None,
               group_labels=True):
    from sklearn.preprocessing import LabelEncoder
    """x must be a numpy array where the first dimension is the number of inputs"""

    if labels is not None:
        # re-associate labels according to the desired rule, recall that each entry of labels array is the list of
        # objects composing that class
        new_y = []
        kept = []
        for j, elem in enumerate(my_y):
            for i, new_label in enumerate(labels):
                if str(elem) in new_label:
                    if group_labels:
                        new_y.append(i)
                    else:
                        new_y.append(elem)
                    kept.append(j)
                    break
        my_y = new_y
        print(f'{len(kept)}/{len(my_x)} recordings kept belonging to {len(labels)} classes')
        my_x = my_x[kept]

    if normalize:
        my_x = (my_x - my_x.min()) / (my_x.max() - my_x.min())
    if sparse_labels:
        label_encoder = LabelEncoder()
        my_y = keras.utils.to_categorical(label_encoder.fit_transform(my_y))
    if shuffle:
        rnd = np.random.permutation(len(my_y))
        my_x = my_x[rnd]
        my_y = my_y[rnd]

    # Checking for duplicates
    u, c = np.unique(my_x, return_counts=True, axis=0)
    dup = u[c > 1]
    if dup.shape[0] != 0:
        print('WARNING: duplicates found!')

    tr_idx = round(len(my_y) * tr_split)
    val_idx = round(len(my_y) * (tr_split + val_split))
    return (my_x[:tr_idx], my_y[:tr_idx]), (my_x[tr_idx:val_idx], my_y[tr_idx:val_idx]), (
        my_x[val_idx:], my_y[val_idx:])


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
    model_to_tune.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])
    learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_val, y_val), batch_size=16)
    learner.lr_find()
    learner.lr_plot()


def training_phase():
    pass


if __name__ == '__main__':
    # np.random.seed(1105)

    """Classification setting: Select the file from which takes measurements, the epoch of interest, and decide which 
    object could be considered as a class: classes is a list, in which each entry is one or more object composing a 
    class (this is still not really clean nor elegant, but the idea is that of grouping objects by shape and reduce 
    the number of classes) """

    FILE = 'MRec40'  # MRec40, ZRec50 or ZRec50_Mini
    PATH = f'../data/Objects Task DL Project/{FILE}.neo.mat'
    EPOCH = 'hold'  # ['start', 'rest', 'motor', 'fixlon', 'fix', 'cue', 'mem', 'react', 'go', 'hold', 'rew', 'intert', 'end']

    selector = ObjectSelector()
    classes = [
        selector.get_shape('mixed'),
        # selector.get_shape('rings'),
        # selector.get_shape('boxes'),
        # selector.get_shape('balls'),
        # selector.get_shape('cubes'),
        #  selector.get_shape('strength'),
        #  selector.get_shape('precision')
    ]
    classes, labels = selector.get_non_special()
    # classes = None

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=f'../log/{datetime.now().strftime("%Y_%m_%d_%H_%M.log")}'
    )

    LR = 7e-3  # None
    K = 10

    """Load the desired measurements, from file or from cache if available"""
    logging.info(f'Loading dataset at {PATH}\nSelecting epoch {EPOCH}')
    try:
        X = np.load(f'temp/{FILE}_{EPOCH}_X.npy')
        Y = np.load(f'temp/{FILE}_{EPOCH}_Y.npy')
        logging.info(f'Windows and objects loaded from chace;\n\tX - {X.shape}\n\tY - {Y.shape}')

    except IOError:
        wrapper = DataWrapper()
        wrapper.load(PATH)
        X, Y, _ = wrapper.get_epochs(EPOCH)
        np.save(f'temp/{FILE}_{EPOCH}_X.npy', X)
        np.save(f'temp/{FILE}_{EPOCH}_Y.npy', Y)
        logging.info('Windows and objects loaded from records;\n')

    # print(f'Loaded {len(Y)} records')
    # objects = set(Y)
    # print(f'{len(objects)} CLASSES:\n\t{objects}')

    """Split the windows and objects lists into train, validation and test set"""
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_sets(X, Y, tr_split=0.1, val_split=0.75,
                                                                      labels=classes,
                                                                      normalize=True,
                                                                      group_labels=True)
    print('Train: ', x_train.shape, y_train.shape)
    print('Validation: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    (_, channels, samples) = x_train.shape
    outputs = y_train.shape[1]

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

    es = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.005, patience=10)
    checkpoint = keras.callbacks.ModelCheckpoint(f'temp/best_model.hdf5', monitor='val_loss',
                                                 save_best_only=True)

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=100, batch_size=16, verbose=1,
                        callbacks=[checkpoint, es])

    # plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label=f'val')

    print(f'Best accuracy: {max(history.history["val_accuracy"])}\n')
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

    #model.set_weights(average_weights(k_weights))
    #model.save_weights('temp/average_weights.hdf5')
    _, accuracy = model.evaluate(x_test, y_test)
    prediction = model.predict(x_test)
    print(f'Test accuracy: {accuracy}')

    conf_matrix = confusion_matrix(y_test.argmax(axis=1), prediction.argmax(axis=1))
    conf_matrix_norm = normalize(conf_matrix, axis=1, norm='l1')
    print(conf_matrix)

    fig, ax = plt.subplots()
    plt.title(f'{FILE}_{EPOCH}_{outputs}_Confusion Matrix [Acc: {accuracy}]')
    plt.imshow(conf_matrix_norm, cmap='Reds')
    plt.colorbar()
    for i in range(outputs):
        for j in range(outputs):
            c = conf_matrix[i][j]
            if c > 0:
                ax.text(j, i, str(c), va='center', ha='center', c='white')
    plt.xticks(ticks=[i for i in range(outputs)], labels=labels)
    plt.yticks(ticks=[i for i in range(outputs)], labels=labels)
    plt.savefig(f'../plots/{FILE}_{EPOCH}_{outputs}_conf_matrix.png')
    plt.show()
