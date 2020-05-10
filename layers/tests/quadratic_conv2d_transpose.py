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

  def test_conv2d_transpose(self):
    """Test if results are same as a normal Conv2D layer"""
    out_channels = 4
    kernel_size = (3, 4)
    strides = (2, 3)

    # define layer
    qconv2d_transpose_layer_object = layers.QuadraticConv2DTranspose(out_channels,
                                                                     kernel_size,
                                                                     strides,
                                                                     use_linear_and_bias=True)
    conv2d_transpose_layer_object = tf.keras.layers.Conv2DTranspose(out_channels,
                                                                    kernel_size,
                                                                    strides,
                                                                    padding='SAME',
                                                                    use_bias=False)

    layer_input = tf.random.uniform(shape=[7, 13, 17, 8], minval=-1., maxval=1.)

    # forced Conv2DTranspose kernel
    kernel_conv2dt = tf.random.uniform(shape=[kernel_size[0], kernel_size[1], out_channels, layer_input.shape[-1]], minval=-1, maxval=1)
    conv2d_transpose_layer_object.add_weight = MagicMock(return_value=kernel_conv2dt)

    # force same kernel into QConv2D: linearize & pad
    kernel_qconv2dt = tf.reverse(kernel_conv2dt, tf.constant([0, 1]))
    kernel_qconv2dt = tf.transpose(kernel_qconv2dt, perm=[0, 1, 3, 2])
    # flatten
    kernel_qconv2dt = tf.reshape(kernel_qconv2dt, shape=[1, 1, kernel_size[0] * kernel_size[1], layer_input.shape[-1], out_channels])
    kernel_qlen = kernel_size[0] * kernel_size[1] * (kernel_size[0] * kernel_size[1] + 1) // 2
    # embed in quadratic kernel
    kernel_qconv2dt = tf.pad(kernel_qconv2dt, tf.constant([[0, 0], [0, 0], [1, kernel_qlen], [0, 0], [0, 0]]))
    qconv2d_transpose_layer_object.add_weight = MagicMock(return_value=kernel_qconv2dt)

    layer_output = qconv2d_transpose_layer_object(layer_input)
    answer = conv2d_transpose_layer_object(layer_input)

    self.assertGreaterEqual(1e-4, tf.reduce_max(tf.abs(layer_output - answer)), "Should be equal")


if __name__ == '__main__':
  unittest.main()