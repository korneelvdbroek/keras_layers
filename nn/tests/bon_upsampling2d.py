import unittest
import tensorflow as tf

import nn
from unittest.mock import MagicMock


class TestQuadraticConv2D(unittest.TestCase):
  @staticmethod
  def get_layer_input():
    layer_input = tf.constant([[1, 2, 3, 4, 5],
                               [4, 5, 6, 7, 8],
                               [7, 8, 9, 10, 11],
                               [1, 2, 3, 4, 5],
                               [2, 3, 4, 5, 6]])
    layer_input = tf.cast(layer_input, dtype=tf.float32)
    layer_input = tf.expand_dims(layer_input, axis=0)
    layer_input = tf.expand_dims(layer_input, axis=-1)
    return layer_input

  def test_equivalence_to_conv2d_transpose(self):
    """Test """
    inputs = self.get_layer_input()

    kernel = tf.constant([[2, 3, 5, 1, 2, 3],
                          [7, 11, 13, 11, 12, 13],
                          [17, 19, 23, 21, 22, 23]])
    kernel = tf.expand_dims(kernel, axis=2)
    kernel = tf.expand_dims(kernel, axis=3)
    kernel = tf.cast(kernel, dtype=tf.float32)
    strides = (2, 1)
    out_channels = kernel.shape[-1]

    # compute via conv2d(bon_upsampling2d())
    upsampled = nn.bon_upsampling2d(inputs, strides)
    tf.print('upsampled = ', upsampled[0, :, :, 0], summarize=100)
    kernel_reverse = tf.reverse(kernel, tf.constant([0, 1]))
    tf.print('upsampled.shape = ', upsampled.shape)
    pad_total = tuple(max(kernel.shape[i] - 1, 0) for i in range(2))
    paddings = [[0, 0]] + [[pad_total[i] - pad_total[i] // 2, (pad_total[i] // 2)] for i in range(2)] + [[0, 0]]
    tf.print('computed padding', paddings)
    tf.print('conv2d padding', nn._compute_padding(upsampled.shape, 1, kernel.shape[0:2], strides))

    upsampled = tf.pad(upsampled, paddings)
    outputs = tf.nn.conv2d(upsampled, kernel_reverse, strides=(1, 1), padding='VALID')
    tf.print('outputs = ', outputs[0, :, :, 0], summarize=100)
    tf.print('outputs shape =', outputs.shape)

    # compute via conv2d_transpose()
    in_channels = inputs.shape[0]
    output_shape = tf.constant([in_channels, inputs.shape[1]*strides[0], inputs.shape[2]*strides[1], out_channels])
    answer = tf.nn.conv2d_transpose(inputs, kernel, output_shape, strides, padding='SAME')
    tf.print('answer = ', answer[0, :, :, 0], summarize=100)

    self.assertGreaterEqual(1e-4, tf.reduce_max(tf.abs(outputs - answer)), "Should be zero")


if __name__ == '__main__':
  unittest.main()
