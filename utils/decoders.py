import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, InputLayer, Reshape, Permute, Dense, Dropout, SimpleRNN, GRU, LSTM, Conv2D, \
    MaxPool2D, Flatten, TimeDistributed, AvgPool2D, Conv1D, AvgPool1D, MultiHeadAttention, LayerNormalization, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.activations import elu, relu
from math import ceil

TEMP_FOLDER = 'D:\\Workspaces\\PycharmProjects\\motor_signal_classification\\classifier\\temp'


class DenseNNClassification:
    """
    Class for the dense (fully-connected) neural network decoder

    Parameters
    ----------
    units: integer or vector of integers, optional, default 400
        This is the number of hidden units in each layer
        If you want a single layer, input an integer (e.g. units=400 will give you a single hidden layer with 400 units)
        If you want multiple layers, input a vector (e.g. units=[400,200]) will give you 2 hidden layers with 400 and 200 units, respectively.
        The vector can either be a list or an array

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out
    """
    def __init__(self, window, channels, outputs, units=400, dropout=0):
        self.name = 'dnn'
        if type(units) is not list:
            units = [units]

        self.model = Sequential()
        # Reshaping the input to fit the feedforward network
        self.model.add(Reshape(target_shape=(channels * window,), input_shape=(channels, window)))
        # Add hidden layers
        for i in range(len(units)):
            if units[i] == 0:
                break
            layer_activation = 'relu' if i == 0 else 'tanh'
            self.model.add(Dense(units[i], activation=layer_activation))
            if dropout != 0:
                self.model.add(Dropout(dropout))
        # Add output layer
        self.model.add(Dense(outputs, activation='softmax'))

        # Fit model (and set fitting parameters)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.initial_weights = self.model.get_weights()

    def fit(self, x_train, y_train, x_val=None, y_val=None, num_epochs=10, batch_size=0, verbose=0):
        """
        Train DenseNN Decoder

        Parameters
        ----------
        x_train: numpy 3d array of shape [n_samples,n_features,n_time_steps]
            This is the neural data.

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted, must be one-hot encoded

        x_val: see x_train
            This is neural data used to validate

        y_val: see y_train

        num_epochs: integer, optional, default 10
            Number of epochs used for training

        batch_size: integer, optional, default 0
            Number of samples used per iteration

        verbose: binary, optional, default=0
            Whether to show progress of the fit after each epoch
        """
        if x_val is not None:
            checkpoint = ModelCheckpoint(f'{TEMP_FOLDER}/{self.name}_best.hdf5', monitor='val_loss',
                                         save_best_only=True)
            history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size,
                           callbacks=[checkpoint],
                           epochs=num_epochs, verbose=verbose)
            self.model.load_weights(f'{TEMP_FOLDER}/{self.name}_best.hdf5')
        else:
            history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose)
        return history

    def predict(self, x_test):
        """
        Predict outcomes using trained DenseNN Decoder

        Parameters
        ----------
        x_test: numpy 3d array of shape [n_samples,n_features,n_time_steps]
            This is the neural data being used to predict outputs.

        Returns
        -------
        prediction: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs one-hot encoded
        """
        prediction = self.model.predict(x_test)
        return prediction

    def reset_weights(self):
        self.model.set_weights(self.initial_weights)


class SimpleRNNClassification:
    """
    Class for the RNN decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out
    """
    def __init__(self, window, channels, outputs, units=400, dropout=0):
        self.name = 'rnn'
        self.model = Sequential()
        # Switching channels and window axis to fit RNN
        self.model.add(Permute((2, 1), input_shape=(channels, window)))
        # Add recurrent layer
        self.model.add(SimpleRNN(units, dropout=dropout, recurrent_dropout=dropout))
        if dropout != 0:
            self.model.add(Dropout(dropout))
        # Add output layer
        self.model.add(Dense(outputs, activation='softmax'))

        # Fit model (and set fitting parameters)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.initial_weights = self.model.get_weights()

    def fit(self, x_train, y_train, x_val=None, y_val=None, num_epochs=10, batch_size=0, verbose=0):
        """
        Train RNN Decoder

        Parameters
        ----------
        x_train: numpy 3d array of shape [n_samples,n_features,n_time_steps]
            This is the neural data.

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted, must be one-hot encoded

        x_val: see x_train
            This is neural data used to validate

        y_val: see y_train

        num_epochs: integer, optional, default 10
            Number of epochs used for training

        batch_size: integer, optional, default 0
            Number of samples used per iteration

        verbose: binary, optional, default=0
            Whether to show progress of the fit after each epoch
        """
        if x_val is not None:
            checkpoint = ModelCheckpoint(f'{TEMP_FOLDER}/{self.name}_best.hdf5', monitor='val_loss',
                                         save_best_only=True)
            history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size,
                           callbacks=[checkpoint], epochs=num_epochs, verbose=verbose)
            self.model.load_weights(f'{TEMP_FOLDER}/{self.name}_best.hdf5')
        else:
            history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose)
        return history

    def predict(self, x_test):
        """
        Predict outcomes using trained RNN Decoder

        Parameters
        ----------
        x_test: numpy 3d array of shape [n_samples,n_features,n_time_steps]
            This is the neural data being used to predict outputs.

        Returns
        -------
        prediction: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs one-hot encoded
        """
        prediction = self.model.predict(x_test)
        return prediction

    def reset_weights(self):
        self.model.set_weights(self.initial_weights)


class GRUClassification:
    """
    Class for the GRU decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out
    """
    def __init__(self, window, channels, outputs, units=400, dropout=0):
        self.name = 'gru'
        self.model = Sequential()
        # Switching channels and window axis to fit RNN
        self.model.add(Permute((2, 1), input_shape=(channels, window)))
        # Add recurrent layer
        self.model.add(GRU(units, dropout=dropout, recurrent_dropout=dropout))
        if dropout != 0:
            self.model.add(Dropout(dropout))
        # Add output layer
        self.model.add(Dense(outputs, activation='softmax'))

        # Fit model (and set fitting parameters)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.initial_weights = self.model.get_weights()

    def fit(self, x_train, y_train, x_val=None, y_val=None, num_epochs=10, batch_size=0, verbose=0):
        """
        Train GRU Decoder

        Parameters
        ----------
        x_train: numpy 3d array of shape [n_samples,n_features,n_time_steps]
            This is the neural data.

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted, must be one-hot encoded

        x_val: see x_train
            This is neural data used to validate

        y_val: see y_train

        num_epochs: integer, optional, default 10
            Number of epochs used for training

        batch_size: integer, optional, default 0
            Number of samples used per iteration

        verbose: binary, optional, default=0
            Whether to show progress of the fit after each epoch
        """
        if x_val is not None:
            checkpoint = ModelCheckpoint(f'{TEMP_FOLDER}/{self.name}_best.hdf5', monitor='val_loss',
                                         save_best_only=True)
            history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size,
                           callbacks=[checkpoint], epochs=num_epochs, verbose=verbose)
            self.model.load_weights(f'{TEMP_FOLDER}/{self.name}_best.hdf5')
        else:
            history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose)

        return history

    def predict(self, x_test):
        """
        Predict outcomes using trained GRU Decoder

        Parameters
        ----------
        x_test: numpy 3d array of shape [n_samples,n_features,n_time_steps]
            This is the neural data being used to predict outputs.

        Returns
        -------
        prediction: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs one-hot encoded
        """
        prediction = self.model.predict(x_test)
        return prediction

    def reset_weights(self):
        self.model.set_weights(self.initial_weights)


class LSTMClassification:
    """
    Class for the LSTM decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out
    """
    def __init__(self, window, channels, outputs, units=400, dropout=0):
        self.name = 'lstm'
        self.model = Sequential()
        # Switching channels and window axis to fit RNN
        self.model.add(Permute((2, 1), input_shape=(channels, window)))
        # Add recurrent layer
        self.model.add(LSTM(units, dropout=dropout, recurrent_dropout=dropout))
        if dropout != 0:
            self.model.add(Dropout(dropout))
        # Add output layer
        self.model.add(Dense(outputs, activation='softmax'))

        # Fit model (and set fitting parameters)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.initial_weights = self.model.get_weights()

    def fit(self, x_train, y_train, x_val=None, y_val=None, num_epochs=10, batch_size=0, verbose=0):
        """
        Train LSTM Decoder

        Parameters
        ----------
        x_train: numpy 3d array of shape [n_samples,n_features,n_time_steps]
            This is the neural data.

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted, must be one-hot encoded

        x_val: see x_train
            This is neural data used to validate

        y_val: see y_train

        num_epochs: integer, optional, default 10
            Number of epochs used for training

        batch_size: integer, optional, default 0
            Number of samples used per iteration

        verbose: binary, optional, default=0
            Whether to show progress of the fit after each epoch
        """
        if x_val is not None:
            checkpoint = ModelCheckpoint(f'{TEMP_FOLDER}/{self.name}_best.hdf5', monitor='val_loss',
                                         save_best_only=True)
            history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size,
                           callbacks=[checkpoint], epochs=num_epochs, verbose=verbose)
            self.model.load_weights(f'{TEMP_FOLDER}/{self.name}_best.hdf5')
        else:
            history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose)
        return history

    def predict(self, x_test):
        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        x_test: numpy 3d array of shape [n_samples,n_features,n_time_steps]
            This is the neural data being used to predict outputs.

        Returns
        -------
        prediction: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs one-hot encoded
        """
        prediction = self.model.predict(x_test)
        return prediction

    def reset_weights(self):
        self.model.set_weights(self.initial_weights)


class CNNClassification:
    """
    Class for the CNN decoder

    Parameters
    ----------
    filters: integer, optional, default 40
        Number of filters applied in the convolutional layer

    size: tuple or integer, optional, default (2,2)
        Shape of each filter

    stride: tuple or integer, optional, default (1, 1)

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    pool_size: decimal, optional, default (2, 2), 0 to avoid this layer
        Shape of pooling layer filter, set 0 to deactivate it
    """
    def __init__(self, window, channels, outputs, filters=40, size=2, dropout=0, pool_size=0):
        self.name = 'cnn'
        self.model = Sequential()
        # Reshaping the input to fit the convolutional network
        self.model.add(Reshape(target_shape=(channels, window, 1), input_shape=(channels, window)))
        # Add convolutional layer
        self.model.add(Conv2D(filters, size, activation='relu'))
        if dropout != 0:
            self.model.add(Dropout(dropout))
        if pool_size != 0:
            self.model.add(MaxPool2D(pool_size))
        self.model.add(Flatten())
        # Add output layer
        self.model.add(Dense(outputs, activation='softmax'))

        # Fit model (and set fitting parameters)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.initial_weights = self.model.get_weights()

    def fit(self, x_train, y_train, x_val=None, y_val=None, num_epochs=10, batch_size=0, verbose=0):
        """
        Train CNN Decoder

        Parameters
        ----------
        x_train: numpy 3d array of shape [n_samples,n_features,n_time_steps]
            This is the neural data.

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted, must be one-hot encoded

        x_val: see x_train
            This is neural data used to validate

        y_val: see y_train

        num_epochs: integer, optional, default 10
            Number of epochs used for training

        batch_size: integer, optional, default 0
            Number of samples used per iteration

        verbose: binary, optional, default=0
            Whether to show progress of the fit after each epoch
        """
        if x_val is not None:
            checkpoint = ModelCheckpoint(f'{TEMP_FOLDER}/{self.name}_best.hdf5', monitor='val_loss',
                                         save_best_only=True)
            history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size,
                           callbacks=[checkpoint], epochs=num_epochs, verbose=verbose)
            self.model.load_weights(f'{TEMP_FOLDER}/{self.name}_best.hdf5')
        else:
            history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose)
        return history

    def predict(self, x_test):
        """
        Predict outcomes using trained CNN Decoder

        Parameters
        ----------
        x_test: numpy 3d array of shape [n_samples,n_features,n_time_steps]
            This is the neural data being used to predict outputs.

        Returns
        -------
        prediction: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs one-hot encoded
        """
        prediction = self.model.predict(x_test)
        return prediction

    def reset_weights(self):
        self.model.set_weights(self.initial_weights)


class EEGNet:
    """
    Class for the simple eeg neural network decoder

    Parameters
    ----------
    units: integer or vector of integers, optional, default 400
        This is the number of hidden units in each layer
        If you want a single layer, input an integer (e.g. units=400 will give you a single hidden layer with 400 units)
        If you want multiple layers, input a vector (e.g. units=[400,200]) will give you 2 hidden layers with 400 and 200 units, respectively.
        The vector can either be a list or an array

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out
    """
    def __init__(self, window, channels, outputs, dropout=0):
        self.name = 'eeg_net'
        self.model = Sequential()
        # Reshaping the input to fit the convolutional network
        self.model.add(Reshape((channels, window, 1), input_shape=(channels, window)))
        # Add hidden layers
        self.model.add(Conv2D(10, (1, 2), padding='same', activation=elu))
        if dropout != 0:
            self.model.add(Dropout(dropout))
        self.model.add(Conv2D(10, (round(channels / 6), 1), strides=(round(channels / 6), 1), activation=elu))
        if dropout != 0:
            self.model.add(Dropout(dropout))
        self.model.add(AvgPool2D((2, 1)))
        self.model.add(Flatten())
        # Add output layer
        self.model.add(Dense(outputs, activation='softmax'))

        # Fit model (and set fitting parameters)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.initial_weights = self.model.get_weights()

    def fit(self, x_train, y_train, x_val=None, y_val=None, num_epochs=10, batch_size=0, verbose=0):
        """
        Train EEGNet Decoder

        Parameters
        ----------
        x_train: numpy 3d array of shape [n_samples,n_features,n_time_steps]
            This is the neural data.

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted, must be one-hot encoded

        x_val: see x_train
            This is neural data used to validate

        y_val: see y_train

        num_epochs: integer, optional, default 10
            Number of epochs used for training

        batch_size: integer, optional, default 0
            Number of samples used per iteration

        verbose: binary, optional, default=0
            Whether to show progress of the fit after each epoch
        """
        if x_val is not None:
            checkpoint = ModelCheckpoint(f'{TEMP_FOLDER}/{self.name}_best.hdf5', monitor='val_loss',
                                         save_best_only=True)
            es = EarlyStopping('val_loss', min_delta=0.01, patience=10)
            history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size,
                           callbacks=[checkpoint, es],
                           epochs=num_epochs, verbose=verbose)
            self.model.load_weights(f'{TEMP_FOLDER}/{self.name}_best.hdf5')
        else:
            history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose)
        return history
    
    def predict(self, x_test):
        """
        Predict outcomes using trained DenseNN Decoder

        Parameters
        ----------
        x_test: numpy 3d array of shape [n_samples,n_features,n_time_steps]
            This is the neural data being used to predict outputs.

        Returns
        -------
        prediction: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs one-hot encoded
        """
        prediction = self.model.predict(x_test)
        return prediction

    def reset_weights(self):
        self.model.set_weights(self.initial_weights)


class EEGNetv2:
    """
    Class for the simple eeg neural network decoder

    Parameters
    ----------
    units: integer or vector of integers, optional, default 400
        This is the number of hidden units in each layer
        If you want a single layer, input an integer (e.g. units=400 will give you a single hidden layer with 400 units)
        If you want multiple layers, input a vector (e.g. units=[400,200]) will give you 2 hidden layers with 400 and 200 units, respectively.
        The vector can either be a list or an array

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out
    """
    def __init__(self, channels, outputs, filters=10, filters_size=None, dropout=0, recurrent_layer='lstm', units=16, neurons=0):
        self.name = 'eeg_net_v2'
        if type(filters) is not list:
            filters = [filters]
        if filters_size is None:
            filters_size = ceil(channels / 6)
        if type(filters_size) is not list:
            filters_size = [filters_size]
        if len(filters) != len(filters_size):
            IOError('Insert same number of filters and sizes')

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(channels, None)))
        # Reshaping the input to fit the recurrent network
        self.model.add(Permute((2, 1)))
        # Reshaping the input to fit the convolutional network
        self.model.add(Reshape((-1, channels, 1)))
        # Extract spatial features
        self.model.add(TimeDistributed(
            Conv1D(filters[0], filters_size[0], strides=filters_size[0], activation=elu)))
        for f, s in zip(filters[1:], filters_size[1:]):
            if f == 0:
                break
            self.model.add(TimeDistributed(Conv1D(f, s, padding='same', activation=elu)))
        if dropout != 0:
            self.model.add(TimeDistributed(Dropout(dropout)))
        self.model.add(TimeDistributed(AvgPool1D(2)))
        self.model.add(TimeDistributed(Flatten()))
        # Extract temporal features
        if recurrent_layer == 'lstm':
            self.model.add(LSTM(units))
        elif recurrent_layer == 'gru':
            self.model.add(GRU(units))
        self.model.add(Flatten())
        # Add output layer
        if neurons != 0:
            self.model.add(Dense(neurons))
        self.model.add(Dense(outputs, activation='softmax'))

        # Fit model (and set fitting parameters)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.initial_weights = self.model.get_weights()

    def fit(self, x_train, y_train, x_val=None, y_val=None, num_epochs=10, batch_size=0, verbose=0):
        """
        Train EEGNet Decoder

        Parameters
        ----------
        x_train: numpy 3d array of shape [n_samples,n_features,n_time_steps]
            This is the neural data.

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted, must be one-hot encoded

        x_val: see x_train
            This is neural data used to validate

        y_val: see y_train

        num_epochs: integer, optional, default 10
            Number of epochs used for training

        batch_size: integer, optional, default 0
            Number of samples used per iteration

        verbose: binary, optional, default=0
            Whether to show progress of the fit after each epoch
        """
        # outputs = y_train.shape[1]
        if x_val is not None:
            checkpoint = ModelCheckpoint(f'{TEMP_FOLDER}/{self.name}_best.hdf5', monitor='val_loss',
                                         save_best_only=True)
            es = EarlyStopping('val_loss', min_delta=0.01, patience=10)
            history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size,
                           callbacks=[checkpoint, es], epochs=num_epochs, verbose=verbose)
            self.model.load_weights(f'{TEMP_FOLDER}/{self.name}_best.hdf5')
        else:
            history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose)

        return history

    def predict(self, x_test):
        """
        Predict outcomes using trained DenseNN Decoder

        Parameters
        ----------
        x_test: numpy 3d array of shape [n_samples,n_features,n_time_steps]
            This is the neural data being used to predict outputs.

        Returns
        -------
        prediction: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs one-hot encoded
        """
        prediction = self.model.predict(x_test)
        return prediction

    def reset_weights(self):
        self.model.set_weights(self.initial_weights)


