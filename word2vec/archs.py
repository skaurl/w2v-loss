from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from metrics import *

weight_decay = 1e-4

def w2v(args):
    input = Input(shape=(4958,))
    x = Dense(args.num_features, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(input)
    output = Dense(4958, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))(x)
    return Model(input, output)

def w2v_arcface(args):
    input = Input(shape=(5164,))
    y = Input(shape=(5164,))
    x = Dense(args.num_features, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(input)
    output = ArcFace(5164, regularizer=regularizers.l2(weight_decay))([x, y])
    return Model([input, y], output)

def w2v_cosface(args):
    input = Input(shape=(4958,))
    y = Input(shape=(4958,))
    x = Dense(args.num_features, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(input)
    output = CosFace(4958, regularizer=regularizers.l2(weight_decay))([x, y])
    return Model([input, y], output)

def w2v_sphereface(args):
    input = Input(shape=(4958,))
    y = Input(shape=(4958,))
    x = Dense(args.num_features, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(input)
    output = SphereFace(4958, regularizer=regularizers.l2(weight_decay))([x, y])
    return Model([input, y], output)