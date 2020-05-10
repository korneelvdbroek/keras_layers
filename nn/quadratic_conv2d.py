# !/usr/bin/env python

"""Definition of a 2D quadratic convolution"""

import math
import tensorflow as tf


def quadratic_conv2d(inputs, out_channels, kernel_size, strides, flat_kernel, use_linear_and_bias=True, padding='same'):
  """
  :param inputs:                 4D input tensor (shape=[batch_size, height, width, in_channels])
  :param out_channels:           number of output channels
  :param kernel_size:            2-tuple of input elements the kernel acts upon
  :param strides:                2-tuple with strides
  :param flat_kernel:            flattened quadratic kernel (shape=[1, 1, kernel_length, in_channels, out_channels]
  :param use_linear_and_bias:    indicates whether a bias term and a linear term (the standard conv2d)
                                 should be included (default = True)
  :param padding:                should be 'SAME' or 'VALID'
  :return:                       4D output tensor (shape=[batch_size, height, width, out_channels])
  """
  if padding.lower() not in ['same', 'valid']:
    raise ValueError("Padding must be 'SAME' or 'VALID' for quadratic_conv2d")

  # check if flat_kernel has right shape
  in_channels = inputs.shape[-1]
  expected_kernel_shape = get_flat_kernel_shape(in_channels, out_channels, kernel_size, use_linear_and_bias)
  if flat_kernel.shape != expected_kernel_shape:
    raise ValueError("flat_kernel should have shape {0} but found shape {1}".format(
      expected_kernel_shape, flat_kernel.shape))

  if padding == 'same':
    inputs = pad_same(inputs, kernel_size, strides)

  return _quadratic_conv2d(inputs, kernel_size, strides, flat_kernel, use_linear_and_bias)


def get_flat_kernel_shape(in_channels, out_channels, kernel_size, use_linear_and_bias):
  flat_kernel_length = (kernel_size[0] * kernel_size[1] * (kernel_size[0] * kernel_size[1] + 1)) // 2
  if use_linear_and_bias:
    flat_kernel_length += kernel_size[0] * kernel_size[1] + 1

  return [1, 1, flat_kernel_length, in_channels, out_channels]


def _quadratic_conv2d(inputs, kernel_size, strides, flat_kernel, use_linear_and_bias):
  ## kernel = [1, 1, w x h, in_ch, out_ch]

  linear_coeff = [tf.ones(shape=[inputs.shape[0],
                                 inputs.shape[1] - kernel_size[0] + 1,
                                 inputs.shape[2] - kernel_size[1] + 1,
                                 inputs.shape[3]], dtype=inputs.dtype)] if use_linear_and_bias else []

  offset_input = linear_coeff + (
    [inputs[:, (i // kernel_size[1]):(i // kernel_size[1]) + inputs.shape[1] - kernel_size[0] + 1,
     (i % kernel_size[1]):(i % kernel_size[1]) + inputs.shape[2] - kernel_size[1] + 1, :]
     for i in range(kernel_size[0] * kernel_size[1])])

  # see what elements need to be multiplied
  include_linear = 1 if use_linear_and_bias else 0
  idxs = [(i, j) for i in range(kernel_size[0] * kernel_size[1] + include_linear) for j in
          range(i, kernel_size[0] * kernel_size[1] + include_linear)]

  # all quadratic terms (pair-wise multiplied elements which are in the kernel-window) will line up on axis=3
  coeff1_heights = tf.stack([offset_input[i] for i, _ in idxs], axis=3)
  coeff2_heights = tf.stack([offset_input[j] for _, j in idxs], axis=3)

  quadratic = coeff1_heights * coeff2_heights

  # could also be written as a matmul, but strides from conv3d is handy
  outputs = tf.nn.conv3d(quadratic, flat_kernel, strides=[1, strides[0], strides[1], 1, 1], padding='VALID')
  outputs = tf.squeeze(outputs, axis=3)  # remove spurious 3rd dimension

  return outputs


def pad_same(inputs, kernel_size, strides):
  """Apply 'SAME' padding for a conv2d layer"""

  input_shape = inputs.shape
  out_shape = [math.ceil(float(input_shape[i + 1]) / float(strides[i])) for i in range(2)]
  pad_total = tuple(
    max(0, (out_shape[i] - 1) * strides[i] + kernel_size[i] - input_shape[i + 1]) for i in range(2))
  # equivalent to:
  # pad_total = tuple(max(kernel_size[i] - 1 - (input_shape[i+1] - 1) % strides[i], 0) for i in range(2))
  # (input_shape[i+1] - 1) % strides[i] is the extra amount of padding we can save at the end!
  paddings = [[0, 0]] + [[pad_total[i] // 2, pad_total[i] - (pad_total[i] // 2)] for i in range(2)] + [[0, 0]]

  outputs = tf.pad(inputs, paddings)

  return outputs
