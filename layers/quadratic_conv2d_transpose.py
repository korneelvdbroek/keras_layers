# !/usr/bin/env python

"""Definition of the transpose quadratic convolution"""

import tensorflow as tf
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


class QuadraticConv2DTranspose(tf.keras.layers.Layer):
  def __init__(self,
               out_channels,
               kernel_size,
               strides=(1, 1),
               padding='same',
               use_linear_and_bias=True,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               kernel_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(QuadraticConv2DTranspose, self).__init__(
        trainable=trainable,
        name=name,
        **kwargs)
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.strides = strides
    if padding.lower() != 'same':
      raise ValueError("Only padding='SAME' is supported for QuadraticConv2DTranspose")
    self.padding = padding
    self.use_linear_and_bias = use_linear_and_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)

  def build(self, input_shape):
    pass

  def call(self, inputs, **kwargs):
    # bed-of-nails

    # then call quadratic_conv2d --> do we make a function nn.quadratic_conv2d to isolate it?

    pass

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
    base_config = super(QuadraticConv2DTranspose, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
