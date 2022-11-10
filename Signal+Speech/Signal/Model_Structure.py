from Constant import *

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras.layers import Input, Dropout, MaxPooling2D, Flatten, Dense, BatchNormalization, LSTM, Activation
from keras.models import Model
from keras.optimizer_v1 import Adam
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def radio2016_model(inshape=RML2016_FEATURE_DIM, class_num=RML2016_CLASS_NUM, lr=1e-4):
    input = Input(shape=inshape)
    # output = LSTM(units=127, activation="tanh", return_sequences=True)(input)
    output = LSTM(units=256, activation="tanh", return_sequences=True)(input)
    output = LSTM(units=64, activation="tanh", return_sequences=True)(output)

    output = Flatten()(output)
    output = Dense(1024, activation=None)(output)
    output = BatchNormalization()(output)
    output = Activation(activation="relu")(output)
    output = Dropout(0.5)(output)

    # output = Dense(128, activation=None)(output)
    # output = BatchNormalization()(output)
    # output = Activation(activation="relu")(output)

    output = Dense(128, activation='relu')(output)
    output = Dropout(0.5)(output)

    output1 = Dense(class_num, activation='relu')(output)
    # output2 = Dense(class_num, activation='linear')(output)
    model = Model(input, output1)
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    model.summary()
    return model


if __name__ == "__main__":
    model = radio2016_model()

    print("end")


