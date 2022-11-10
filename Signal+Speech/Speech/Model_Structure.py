from Constant import *
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, LSTM
from keras.models import Model
from keras.optimizer_v1 import Adam
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def vctk_model(inshape=VCTK_FEATURE_DIM, class_num=VCTK_CLASS_NUM, lr=1e-3):
    input = Input(shape=inshape)
    output = LSTM(units=32, activation="tanh", return_sequences=True)(input)
    output = LSTM(units=8, activation="tanh", return_sequences=True)(output)
    output = Flatten()(output)
    output = Dense(64, activation='relu')(output)
    output = Dense(class_num, activation='softmax')(output)
    model = Model(input, output)
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


if __name__ == "__main__":
    model = vctk_model()

    print("end")

