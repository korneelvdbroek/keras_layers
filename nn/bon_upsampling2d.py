# !/usr/bin/env python

"""Definition of a transposed 2D quadratic convolution using a bed-of-nails up-sampling"""

import tensorflow as tf


def bon_upsampling2d(inputs, strides, kernel_shape):
  """
  2D bed-of-nails up-sampling. Combining this up-sampling with a conv2d layer is equivalent to conv2d_transpose
  Output is padded such that:
     conv2d(output, padding='VALID') = conv2d_transpose(inputs)
  when one uses a reversed kernel (see tests)

  :param inputs:     4D input tensor (shape=[batch_size, height, width, in_channels])
  :param strides:    2-tuple with strides
  :return:           4D input tensor (shape=[batch_size, height * strides[0], width * strides[1], in_channels])
  """
  pad_total = tuple(kernel_shape[i] - 1 for i in range(2))
  padding_magic = [(1 - (pad_total[i] % 2)) * (1 - (strides[i] % 2)) - max(0, strides[i] - pad_total[i]) // 2 + (
                   pad_total[i] - (pad_total[i] // 2)) for i in range(2)]

  # bed-of-nails up-sampling
  s_x = inputs.shape
  x = tf.reshape(inputs, [-1, s_x[1], 1, s_x[2], 1, s_x[3]])
  padding_height, padding_width = [[min(0, padding_magic[i]) + strides[i] - 1 - (strides[i] // 2),
                                    -min(0, padding_magic[i]) + strides[i] // 2] for i in range(2)]
  bed_of_nails_paddings = tf.constant([[0, 0]]*2 + [padding_height] + [[0, 0]] + [padding_width] + [[0, 0]])
  x = tf.pad(x, bed_of_nails_paddings)
  outputs = tf.reshape(x, [-1, s_x[1] * strides[0], s_x[2] * strides[1], s_x[3]])

  # pad such that output can directly be use in conv2d(padding='VALID')
  conv2d_paddings = [[0, 0]] + [[max(0, padding_magic[i]), - max(0, padding_magic[i]) + pad_total[i]] for i in range(2)] + [[0, 0]]
  outputs = tf.pad(outputs, conv2d_paddings)

  return outputs
