import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Reshape, Activation, Concatenate
from metrics import *

weight_decay = 1e-4

def w2v(args):
    with tf.device('/device:GPU:0'):
        w_inputs = Input(shape=(1,), dtype='int32')
        word_embedding = Embedding(52203, args.num_features)(w_inputs)

        c_inputs = Input(shape=(1,), dtype='int32')
        context_embedding = Embedding(52203, args.num_features)(c_inputs)

        dot_product = Dot(axes=2)([word_embedding, context_embedding])
        dot_product = Reshape((1,), input_shape=(1, 1))(dot_product)
        output = Activation('sigmoid')(dot_product)

    return Model(inputs=[w_inputs, c_inputs], outputs=output)

def w2v_adacos(args):
    with tf.device('/device:GPU:0'):
        w_inputs = Input(shape=(1,), dtype='int32')
        word_embedding = Embedding(52203, args.num_features)(w_inputs)

        c_inputs = Input(shape=(1,), dtype='int32')
        context_embedding = Embedding(52203, args.num_features)(c_inputs)

        dot_product = Dot(axes=2)([word_embedding, context_embedding])
        dot_product_0 = Reshape((1,), input_shape=(1, 1))(dot_product)
        dot_product_0 = 1 - dot_product_0
        dot_product_1 = Reshape((1,), input_shape=(1, 1))(dot_product)
        dot_product = Concatenate(axis=1)([dot_product_0, dot_product_1])

        labels = Input(shape=(2,), dtype='float32')

        output = AdaCos(2, regularizer=regularizers.l2(weight_decay))([dot_product, labels])

    return Model(inputs=[w_inputs, c_inputs, labels], outputs=output)

def w2v_arcface(args):
    with tf.device('/device:GPU:0'):
        w_inputs = Input(shape=(1,), dtype='int32')
        word_embedding = Embedding(52203, args.num_features)(w_inputs)

        c_inputs = Input(shape=(1,), dtype='int32')
        context_embedding = Embedding(52203, args.num_features)(c_inputs)

        dot_product = Dot(axes=2)([word_embedding, context_embedding])
        dot_product_0 = Reshape((1,), input_shape=(1, 1))(dot_product)
        dot_product_0 = 1 - dot_product_0
        dot_product_1 = Reshape((1,), input_shape=(1, 1))(dot_product)
        dot_product = Concatenate(axis=1)([dot_product_0, dot_product_1])

        labels = Input(shape=(2,), dtype='float32')

        output = ArcFace(2, regularizer=regularizers.l2(weight_decay))([dot_product, labels])

    return Model(inputs=[w_inputs, c_inputs, labels], outputs=output)

def w2v_cosface(args):
    with tf.device('/device:GPU:0'):
        w_inputs = Input(shape=(1,), dtype='int32')
        word_embedding = Embedding(52203, args.num_features)(w_inputs)

        c_inputs = Input(shape=(1,), dtype='int32')
        context_embedding = Embedding(52203, args.num_features)(c_inputs)

        dot_product = Dot(axes=2)([word_embedding, context_embedding])
        dot_product_0 = Reshape((1,), input_shape=(1, 1))(dot_product)
        dot_product_0 = 1 - dot_product_0
        dot_product_1 = Reshape((1,), input_shape=(1, 1))(dot_product)
        dot_product = Concatenate(axis=1)([dot_product_0, dot_product_1])

        labels = Input(shape=(2,), dtype='float32')

        output = CosFace(2, regularizer=regularizers.l2(weight_decay))([dot_product, labels])

    return Model(inputs=[w_inputs, c_inputs, labels], outputs=output)

def w2v_sphereface(args):
    with tf.device('/device:GPU:0'):
        w_inputs = Input(shape=(1,), dtype='int32')
        word_embedding = Embedding(52203, args.num_features)(w_inputs)

        c_inputs = Input(shape=(1,), dtype='int32')
        context_embedding = Embedding(52203, args.num_features)(c_inputs)

        dot_product = Dot(axes=2)([word_embedding, context_embedding])
        dot_product_0 = Reshape((1,), input_shape=(1, 1))(dot_product)
        dot_product_0 = 1 - dot_product_0
        dot_product_1 = Reshape((1,), input_shape=(1, 1))(dot_product)
        dot_product = Concatenate(axis=1)([dot_product_0, dot_product_1])

        labels = Input(shape=(2,), dtype='float32')

        output = SphereFace(2, regularizer=regularizers.l2(weight_decay))([dot_product, labels])

    return Model(inputs=[w_inputs, c_inputs, labels], outputs=output)
