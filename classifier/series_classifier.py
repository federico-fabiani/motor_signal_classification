# tensorflow-gpu version 2.4.1
# CUDA version 11.0
# Cudnn version 8.0.5
# Python 3.8

from tensorflow import keras
import numpy as np
import ktrain
from utils.data_processing import *
from utils.object_selector import *
import matplotlib.pyplot as plt


def split_sets(my_x, my_y, tr_split, val_split, shuffle=True, normalize=True, sparse_labels=True, labels=None):
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
                    new_y.append(i)
                    kept.append(j)
                    break
        my_y = new_y
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
    tr_idx = round(len(my_y) * tr_split)
    val_idx = round(len(my_y) * (tr_split + val_split))
    return (my_x[:tr_idx], my_y[:tr_idx]), (my_x[tr_idx:val_idx], my_y[tr_idx:val_idx]), (my_x[val_idx:], my_y[val_idx:])


def EEGNet(features, window, classes, lstm=False):
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
    eeg_net.add(keras.layers.Dense(classes, activation='softmax'))
    return eeg_net


if __name__ == '__main__':
    # np.random.seed(1105)
    FILE = 'MRec40'  # MRec40, ZRec50 or ZRec50_Mini
    PATH = f'../data/Objects Task DL Project/{FILE}.neo.mat'
    EPOCH = 'hold'

    LR = 8e-3  # None

    try:
        X = np.load(f'temp/{FILE}_{EPOCH}_X.npy')
        Y = np.load(f'temp/{FILE}_{EPOCH}_Y.npy')
        print('dataset loaded from cache')

    except IOError:
        wrapper = DataWrapper()
        wrapper.load(PATH)
        X, Y, _ = wrapper.get_epochs(EPOCH)
        np.save(f'temp/{FILE}_{EPOCH}_X.npy', X)
        np.save(f'temp/{FILE}_{EPOCH}_Y.npy', Y)

    print(f'Loaded {len(Y)} records')
    objects = set(Y)
    print(f'{len(objects)} CLASSES:\n\t{objects}')

    selector = ObjectSelector()
    # classes is a list, in which each entry is one or more object composing a class
    # classes = [
    #     selector.get_shape('rings'),
    #     selector.get_shape('boxes')
    # ]
    classes = selector.get_shape('mixed')
    # classes = selector.get_non_special()
    # classes = selector.get_shape('cubes')
    # classes = [
    #     selector.get_shape('strength'),
    #     selector.get_shape('precision')
    # ]
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_sets(X, Y, tr_split=0.7, val_split=0.15, labels=classes)

    # data should be shaped as 196 channels x 100 samples x 3 units. Each entry is then the number of activation of a
    # specific neural unit (int [0-10]), in that window
    print('Train: ', x_train.shape, y_train.shape)
    print('Validation: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    (_, channels, samples) = x_train.shape
    outputs = y_train.shape[1]

    model = EEGNet(channels, samples, outputs)
    model.summary()

    # The following code is necessary to print the model summary on a txt file
    # with open('report.txt', 'w') as fh:
    #     # Pass the file handle in as a lambda function to make it callable
    #     model.summary(print_fn=lambda x: fh.write(x + '\n'))

    if LR is None:
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])
        learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_val, y_val), batch_size=16)
        learner.lr_find()
        learner.lr_plot()

    else:
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=LR), metrics=['accuracy'])
        es = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=5)
        checkpoint = keras.callbacks.ModelCheckpoint(f'temp/best_model.hdf5', monitor='val_accuracy', save_best_only=True)
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=8, verbose=1,
                            callbacks=[checkpoint, es])
        plt.figure()
        plt.title(f'{outputs} Classes Training')
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='val')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.tight_layout()
        plt.legend()
        # plt.savefig(f'../plots/{outputs}_windows_accuracy_training.png')
        plt.show()

        model.load_weights(f'temp/best_model.hdf5')
        _, accuracy = model.evaluate(x_test, y_test)
        print(f'Test accuracy: {accuracy}')
