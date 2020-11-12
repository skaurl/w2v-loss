from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import Input, MaxPooling2D, Dropout, Flatten
from metrics import *

weight_decay = 1e-4

def w2v(args):
    input = Input(shape=(3239, 1, 1))
    x = input
    x = Flatten(input_shape=(3239, 1, 1))(x)
    x = Dense(args.num_features, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
    output = Dense(3239, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))(x)
    return Model(input, output)

def w2v_arcface(args):
    input = Input(shape=(3239, 1, 1))
    y = Input(shape=(3239,))
    x = input
    x = Flatten(input_shape=(3239, 1, 1))(x)
    x = Dense(args.num_features, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
    output = ArcFace(3239, regularizer=regularizers.l2(weight_decay))([x, y])
    return Model([input, y], output)

def w2v_cosface(args):
    input = Input(shape=(3239, 1, 1))
    y = Input(shape=(3239,))
    x = input
    x = Flatten(input_shape=(3239, 1, 1))(x)
    x = Dense(args.num_features, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
    output = CosFace(3239, regularizer=regularizers.l2(weight_decay))([x, y])
    return Model([input, y], output)

def w2v_sphereface(args):
    input = Input(shape=(3239, 1, 1))
    y = Input(shape=(3239,))
    x = input
    x = Flatten(input_shape=(3239, 1, 1))(x)
    x = Dense(args.num_features, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
    output = SphereFace(3239, regularizer=regularizers.l2(weight_decay))([x, y])
    return Model([input, y], output)