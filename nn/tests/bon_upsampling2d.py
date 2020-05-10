import unittest
import tensorflow as tf

import nn


class TestQuadraticConv2D(unittest.TestCase):
  def test_equivalence_to_conv2d_transpose(self):
    """Test that conv2d(bon_upsampling2d()) = conv2d_transpose()"""
    inputs = tf.random.uniform(shape=[1, 10, 23, 8], minval=-10., maxval=10., dtype=tf.float32)

    kernel = tf.random.uniform(shape=[6, 16, 4, inputs.shape[-1]], minval=-10., maxval=10., dtype=tf.float32)
    out_channels = kernel.shape[-2]

    for stride_width in range(1, 20):
      strides = (2, stride_width)

      # compute via conv2d(bon_upsampling2d())
      upsampled = nn.bon_upsampling2d(inputs, strides, kernel.shape)
      kernel_reverse = tf.reverse(kernel, tf.constant([0, 1]))
      kernel_reverse = tf.transpose(kernel_reverse, perm=[0, 1, 3, 2])
      outputs = tf.nn.conv2d(upsampled, kernel_reverse, strides=(1, 1), padding='VALID')

      # compute via conv2d_transpose()
      output_shape = tf.constant([inputs.shape[0], inputs.shape[1]*strides[0], inputs.shape[2]*strides[1], out_channels])
      answer = tf.nn.conv2d_transpose(inputs, kernel, output_shape, strides, padding='SAME')

      self.assertGreaterEqual(5e-3, tf.reduce_max(tf.abs(outputs - answer)), "Should be zero")


if __name__ == '__main__':
  unittest.main()
