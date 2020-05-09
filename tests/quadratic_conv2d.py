import unittest
import tensorflow as tf
import layers
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

  def test_bias(self):
    """Test bias"""
    out_channels = 1

    # define layer
    layer_object = layers.QuadraticConv2D(out_channels=out_channels,
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          use_linear_and_bias=True)

    layer_input = self.get_layer_input()
    in_channels = layer_input.shape[-1]

    # force a kernel into the layer
    kernel_length = 3 * 3 * (3 * 3 + 1) // 2
    kernel_length += 3 * 3 + 1
    kernel_data = [[[1 if (i == 0) else 0]] for i in range(kernel_length)]
    kernel = tf.constant([[kernel_data]], dtype=tf.float32)
    kernel = tf.tile(kernel, multiples=tf.constant([1, 1, 1, in_channels, out_channels], tf.int32))
    layer_object.add_weight = MagicMock(return_value=kernel)

    layer_output = layer_object(layer_input)

    answer = tf.ones(shape=layer_input.shape, dtype=tf.float32)
    self.assertEqual(tf.reduce_max(tf.abs(layer_output - answer)), 0., "Should be equal")

  def test_conv2d(self):
    """Test if results are same as a normal Conv2D layer"""
    out_channels = 5

    # define layer
    qconv2d_layer_object = layers.QuadraticConv2D(out_channels=out_channels,
                                                  kernel_size=(3, 3),
                                                  strides=(1, 1),
                                                  use_linear_and_bias=True)
    conv2d_layer_object = tf.keras.layers.Conv2D(out_channels,
                                                 kernel_size=(3, 3),
                                                 strides=(1, 1),
                                                 use_bias=False,
                                                 padding='same')

    layer_input = tf.random.uniform(shape=[7, 13, 17, 31], minval=-10., maxval=10.)

    # forced Conv2D kernel
    kernel_size = (3, 3)
    kernel_conv2d = tf.random.uniform(shape=[kernel_size[0], kernel_size[1], layer_input.shape[-1], out_channels], minval=-1, maxval=1)
    conv2d_layer_object.add_weight = MagicMock(return_value=kernel_conv2d)

    # force same kernel into QConv2D: linearize & pad
    kernel_qconv2d = tf.reshape(kernel_conv2d, shape=[1, 1, kernel_size[0] * kernel_size[1], layer_input.shape[-1], out_channels])
    kernel_qlen = kernel_size[0] * kernel_size[1] * (kernel_size[0] * kernel_size[1] + 1) // 2
    kernel_qconv2d = tf.pad(kernel_qconv2d, tf.constant([[0, 0], [0, 0], [1, kernel_qlen], [0, 0], [0, 0]]))
    tf.print('qconv2d kernel shape = ', tf.shape(kernel_qconv2d))
    qconv2d_layer_object.add_weight = MagicMock(return_value=kernel_qconv2d)

    layer_output = qconv2d_layer_object(layer_input)
    answer = conv2d_layer_object(layer_input)

    self.assertGreaterEqual(1e-4, tf.reduce_max(tf.abs(layer_output - answer)), "Should be equal")

if __name__ == '__main__':
  unittest.main()