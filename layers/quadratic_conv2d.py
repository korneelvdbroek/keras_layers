# !/usr/bin/env python

"""Definition of a 2D quadratic convolution keras Layer"""

import tensorflow as tf
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

import nn


class QuadraticConv2D(tf.keras.layers.Layer):
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
    super(QuadraticConv2D, self).__init__(
        trainable=trainable,
        name=name,
        **kwargs)
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.use_linear_and_bias = use_linear_and_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)

  def build(self, input_shape):
    in_channels = input_shape[3]

    flat_kernel_length = nn.get_flat_kernel_shape(in_channels, self.out_channels,
                                                  self.kernel_size, self.use_linear_and_bias)
    flat_kernel_shape = [1, 1, flat_kernel_length, in_channels, self.out_channels]

    self.flat_kernel = self.add_weight(
        name='kernel',
        shape=flat_kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)

  def call(self, inputs, **kwargs):
    return nn.quadratic_conv2d(inputs, self.out_channels, self.kernel_size, self.strides, self.flat_kernel,
                               self.use_linear_and_bias, self.padding)

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
