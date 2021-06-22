# tensorflow-gpu version 2.4.1
# CUDA version 11.0
# Cudnn version 8.0.5
# Python 3.8

from classifier.offline_decoder import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    # np.random.seed(1805)
    # logging.basicConfig(
    #     level=logging.getLevelName("INFO"),
    #     format='%(message)s',
    #     filename=f'../log/{datetime.now().strftime("%Y_%m_%d_%H_%M.log")}'
    # )
    # clean_logs('../log', keep=5)

    """Classification setting: Select the file from which takes measurements, the epoch of interest, and decide which 
    object could be considered as a class: classes is a list, in which each entry is one or more object composing a 
    class (this is still not really clean nor elegant, but the idea is that of grouping objects by shape and reduce 
    the number of classes) """

    FILE = 'ZRec50_Mini'  # MRec40, ZRec50 or ZRec50_Mini
    PATH = f'../data/Objects Task DL Project/{FILE}.neo.mat'
    BIN_SIZE = 40  # Milliseconds sampled per time
    LR = 7e-3  # None
    K = 10

    wrapper = DataWrapper()
    wrapper.load(PATH)
    X, Y = wrapper.get_epoch_plus_noise()
    # Normalize X
    X = (X - X.min()) / (X.max() - X.min())
    # Encode labels
    one_hot_encoder = LabelEncoder()
    Y = keras.utils.to_categorical(one_hot_encoder.fit_transform(Y))
    # Shuffle
    rnd = np.random.permutation(len(Y))
    X = X[rnd]
    Y = Y[rnd]

    tr_idx = round(len(Y) * 0.6)
    val_idx = round(len(Y) * (0.6 + 0.2))

    (_, channels, samples) = X.shape
    outputs = Y.shape[1]

    model = keras.models.Sequential()
    model.add(keras.layers.Permute((2, 1), input_shape=(channels, samples)))
    model.add(keras.layers.Reshape((samples, channels, 1)))
    model.add(keras.layers.Conv2D(40, (3, 1), padding='same', activation=keras.activations.elu))
    model.add(keras.layers.Conv2D(40, (1, round(channels / 6)), activation=keras.activations.elu))
    model.add(keras.layers.AvgPool2D((2, 1)))
    if False:
        model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
        model.add(keras.layers.LSTM(32, activation='tanh', return_sequences=True))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(outputs, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=LR), metrics=['accuracy'])

    # es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=5)
    checkpoint = keras.callbacks.ModelCheckpoint(f'temp/best_model.hdf5', monitor='val_loss',
                                                 save_best_only=True)

    history = model.fit(X[:tr_idx], Y[:tr_idx], validation_data=(X[tr_idx:val_idx], Y[tr_idx:val_idx]),
                        epochs=30, batch_size=16, verbose=1,
                        callbacks=[checkpoint])

    plt.figure()
    plt.title(f'{FILE}_{outputs}_Training')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label=f'val')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.tight_layout()
    plt.legend()
    # plt.savefig(f'../plots/{FILE}_{EPOCH}_{outputs}_Training.png')
    plt.show()

    model.load_weights(f'temp/best_model.hdf5')
    print(model.evaluate(X[val_idx:], Y[val_idx:]))
    # prediction = model.predict(x_test)
    # confusion_matrix(prediction, y_test, one_hot_encoder=label_encoder, plot=True)
    # accuracy = accuracy_score(y_true=y_test.argmax(axis=1), y_pred=prediction.argmax(axis=1))
    # print(accuracy)



