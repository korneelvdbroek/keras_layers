# !/usr/bin/env python

"""Definition of quadratic convolution"""

import math
import tensorflow as tf
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


class QuadraticConv2D(tf.keras.layers.Layer):
  def __init__(self,
               out_channels,
               kernel_size,
               strides=(1, 1),
               padding='same',
               use_linear_and_bias=True,
               trainable=True,
               name=None,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               kernel_constraint=None,
               **kwargs):
    super(QuadraticConv2D, self).__init__(
        trainable=trainable,
        name=name,
        **kwargs)
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.strides = strides
    if padding.lower() != 'same':
      raise ValueError("Only padding='SAME' is supported for QuadraticConv2D")
    self.padding = padding
    self.use_linear_and_bias = use_linear_and_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)


  def build(self, input_shape):
    self.batch_size, self.in_height, self.in_width, self.in_channels = input_shape

    # padding='SAME'
    self.out_shape = [self.batch_size] + [math.ceil(float(input_shape[i + 1]) / float(self.strides[i])) for i in range(2)] + [self.out_channels]
    self.pad_total = tuple(max((self.out_shape[i + 1] - 1) * self.strides[i] + self.kernel_size[i] - input_shape[i + 1], 0) for i in range(2))
    self.paddings = [[0, 0]] + [[self.pad_total[i] // 2, self.pad_total[i] - (self.pad_total[i] // 2)] for i in range(2)] + [[0, 0]]

    self.linear_coeff = [tf.ones(input_shape, dtype=self.dtype)] if self.use_linear_and_bias else []

    kernel_length = (self.kernel_size[0] * self.kernel_size[1] * (self.kernel_size[0] * self.kernel_size[1] + 1)) // 2
    if self.use_linear_and_bias:
      kernel_length += self.kernel_size[0] * self.kernel_size[1] + 1

    kernel_shape = [1, 1, kernel_length, self.in_channels, self.out_channels]

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)

  def call(self, inputs, **kwargs):
    input_pad = tf.pad(inputs, self.paddings)

    offset_input = self.linear_coeff + [input_pad[:, (i // self.kernel_size[1]):(i // self.kernel_size[1]) + self.in_height,
                                                     (i % self.kernel_size[1]):(i % self.kernel_size[1]) + self.in_width, :]
                                        for i in range((self.pad_total[0] + 1) * (self.pad_total[1] + 1))]

    # see what elements need to be multiplied
    include_linear = 1 if self.use_linear_and_bias else 0
    idxs = [(i, j) for i in range(self.kernel_size[0] * self.kernel_size[1] + include_linear) for j in
            range(i, self.kernel_size[0] * self.kernel_size[1] + include_linear)]

    # all quadratic terms (pair-wise multiplied elements which are in the kernel-window) will line up on axis=3
    coeff1_heights = tf.stack([offset_input[i] for i, _ in idxs], axis=3)
    coeff2_heights = tf.stack([offset_input[j] for _, j in idxs], axis=3)

    quadratic = coeff1_heights * coeff2_heights

    # could also be written as a matmul, but strides from conv3d is handy
    outputs = tf.nn.conv3d(quadratic, self.kernel, strides=[1, self.strides[0], self.strides[1], 1, 1], padding='VALID')
    outputs = tf.reshape(outputs, shape=self.out_shape)  # remove spurious 3rd dimension

    return outputs

  def get_config(self):
    config = {
        'out_channels': self.out_channels,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'use_linear_and_bias': self.use_linear_and_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint)
    }
    base_config = super(QuadraticConv2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))