# MIT License
#
# Copyright (c) 2018 Hao-Wen Dong and Wen-Yi Hsiao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tensorflow ops."""
import tensorflow as tf

CONV_KERNEL_INITIALIZER = tf.truncated_normal_initializer(stddev=0.05)
DENSE_KERNEL_INITIALIZER = tf.truncated_normal_initializer(stddev=0.05)

dense = lambda i, u: tf.layers.dense(
    i, u, kernel_initializer=DENSE_KERNEL_INITIALIZER)
conv2d = lambda i, f, k, s: tf.layers.conv2d(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)
conv3d = lambda i, f, k, s: tf.layers.conv3d(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)
tconv2d = lambda i, f, k, s: tf.layers.conv2d_transpose(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)
tconv3d = lambda i, f, k, s: tf.layers.conv3d_transpose(
    i, f, k, s, kernel_initializer=CONV_KERNEL_INITIALIZER)

def get_normalization(norm_type, training=None):
    """Return the normalization function."""
    if norm_type == 'batch_norm':
        return lambda x: tf.layers.batch_normalization(x, training=training)
    if norm_type == 'layer_norm':
        return tf.contrib.layers.layer_norm
    if norm_type is None or norm_type == '':
        return lambda x: x
    raise ValueError("Unrecognizable normalization type.")
