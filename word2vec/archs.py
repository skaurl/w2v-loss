from keras.models import Model
from keras.layers import Input, Embedding, Dot, Reshape, Activation
from metrics import *

weight_decay = 1e-4

def w2v(args):
    w_inputs = Input(shape=(1,), dtype='int32')
    word_embedding = Embedding(52203, args.num_features)(w_inputs)

    c_inputs = Input(shape=(1,), dtype='int32')
    context_embedding = Embedding(52203, args.num_features)(c_inputs)

    dot_product = Dot(axes=2)([word_embedding, context_embedding])
    dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)
    output = Activation('sigmoid')(dot_product)

    return Model(inputs=[w_inputs, c_inputs], outputs=output)

def w2v_arcface(args):
    w_inputs = Input(shape=(1,), dtype='int32')
    word_embedding = Embedding(52203, args.num_features)(w_inputs)

    c_inputs = Input(shape=(1,), dtype='int32')
    context_embedding = Embedding(52203, args.num_features)(c_inputs)

    labels = Input(shape=(1,), dtype='float32')

    dot_product = Dot(axes=2)([word_embedding, context_embedding])
    dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)

    output = ArcFace(1, regularizer=regularizers.l2(weight_decay))([dot_product, labels])
    return Model(inputs=[w_inputs, c_inputs, labels], outputs=output)

def w2v_cosface(args):
    w_inputs = Input(shape=(1,), dtype='int32')
    word_embedding = Embedding(52203, args.num_features)(w_inputs)

    c_inputs = Input(shape=(1,), dtype='int32')
    context_embedding = Embedding(52203, args.num_features)(c_inputs)

    labels = Input(shape=(1,), dtype='float32')

    dot_product = Dot(axes=2)([word_embedding, context_embedding])
    dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)

    output = CosFace(1, regularizer=regularizers.l2(weight_decay))([dot_product, labels])
    return Model(inputs=[w_inputs, c_inputs, labels], outputs=output)

def w2v_sphereface(args):
    w_inputs = Input(shape=(1,), dtype='int32')
    word_embedding = Embedding(52203, args.num_features)(w_inputs)

    c_inputs = Input(shape=(1,), dtype='int32')
    context_embedding = Embedding(52203, args.num_features)(c_inputs)

    labels = Input(shape=(1,), dtype='float32')

    dot_product = Dot(axes=2)([word_embedding, context_embedding])
    dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)

    output = SphereFace(1, regularizer=regularizers.l2(weight_decay))([dot_product, labels])
    return Model(inputs=[w_inputs, c_inputs, labels], outputs=output)