import math
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import Constant
from tensorflow.python.keras.utils import tf_utils
import tensorflow_probability as tfp

def _resolve_training(layer, training):
    if training is None:
        training = K.learning_phase()
    if isinstance(training, int):
        training = bool(training)
    if not layer.trainable:
        training = False
    return tf_utils.constant_value(training)

class AdaCos(Layer):
    def __init__(self, num_classes=2, is_dynamic=True, regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self._n_classes = num_classes
        self._init_s = math.sqrt(2) * math.log(num_classes - 1)
        self._is_dynamic = is_dynamic
        self._regularizer = regularizer

    def build(self, input_shape):
        embedding_shape, label_shape = input_shape
        self._w = self.add_weight(shape=(embedding_shape[-1], self._n_classes),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  regularizer=self._regularizer)
        if self._is_dynamic:
            self._s = self.add_weight(shape=(),
                                      initializer=Constant(self._init_s),
                                      trainable=False,
                                      aggregation=tf.VariableAggregation.MEAN)

    def call(self, inputs, training=None):
        x, y = inputs
        x = tf.nn.l2_normalize(x, axis=1)
        w = tf.nn.l2_normalize(self._w, axis=0)
        logits = tf.matmul(x, w)
        is_dynamic = tf_utils.constant_value(self._is_dynamic)
        if not is_dynamic:
            output = tf.multiply(self._init_s, logits)
            return output
        training = _resolve_training(self, training)
        if not training:
            return self._s * logits
        else:
            theta = tf.math.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            b_avg = tf.where(y < 1.0, tf.exp(self._s * logits), tf.zeros_like(logits))
            b_avg = tf.reduce_mean(tf.reduce_sum(b_avg, axis=1))
            theta_class = tf.gather(theta, tf.cast(y, tf.int32))
            theta_med = tfp.stats.percentile(theta_class, q=50)
            self._s.assign(tf.math.log(b_avg) / tf.math.cos(tf.minimum(math.pi / 4, theta_med)))
            self._s.assign_add(-0.5)
            logits *= self._s
            out = tf.nn.sigmoid(logits)
            return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

class ArcFace(Layer):
    def __init__(self, n_classes=2, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = x @ W
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        logits = logits * (1 - y) + target_logits * y
        logits *= self.s
        out = tf.nn.sigmoid(logits)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

class SphereFace(Layer):
    def __init__(self, n_classes=2, s=30.0, m=1.35, regularizer=None, **kwargs):
        super(SphereFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(SphereFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = x @ W
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(self.m * theta)
        logits = logits * (1 - y) + target_logits * y
        logits *= self.s
        out = tf.nn.sigmoid(logits)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

class CosFace(Layer):
    def __init__(self, n_classes=2, s=30.0, m=0.35, regularizer=None, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(CosFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = x @ W
        target_logits = logits - self.m
        logits = logits * (1 - y) + target_logits * y
        logits *= self.s
        out = tf.nn.sigmoid(logits)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)
