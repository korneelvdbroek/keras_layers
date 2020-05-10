# !/usr/bin/env python

"""Definition of a transposed 2D quadratic convolution using a bed-of-nails up-sampling"""

import tensorflow as tf


def bon_upsampling2d(inputs, strides):
  """
  2D bed-of-nails up-sampling. Combining this up-sampling with a conv2d layer is equivalent to conv2d_transpose

  :param inputs:     4D input tensor (shape=[batch_size, height, width, in_channels])
  :param strides:    2-tuple with strides
  :return:           4D input tensor (shape=[batch_size, height * strides[0], width * strides[1], in_channels])
  """
  s_x = inputs.shape
  x = tf.reshape(inputs, [-1, s_x[1], 1, s_x[2], 1, s_x[3]])

  padding_height, padding_width = [[strides[i] - 1, 0]
                                   for i in range(2)]
  print('strides = {0} ==> padding = {1}, {2}'.format(strides, padding_height, padding_width))

  paddings = tf.constant([[0, 0]]*2 + [padding_height] + [[0, 0]] + [padding_width] + [[0, 0]])
  x = tf.pad(x, paddings)
  outputs = tf.reshape(x, [-1, s_x[1] * strides[0], s_x[2] * strides[1], s_x[3]])

  return outputs
