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

"""This file defines the network architecture for the generator."""
import tensorflow as tf
from tensorflow.nn import relu, leaky_relu, tanh, sigmoid
from ..ops import tconv3d, get_normalization

NORMALIZATION = 'batch_norm' # 'batch_norm', 'layer_norm'
ACTIVATION = tanh # relu, leaky_relu, tanh, sigmoid

class Generator:
    def __init__(self, n_tracks, name='Generator'):
        self.n_tracks = n_tracks
        self.name = name

    def __call__(self, tensor_in, condition=None, training=None, slope=None):
        norm = get_normalization(NORMALIZATION, training)
        tconv_layer = lambda i, f, k, s: ACTIVATION(norm(tconv3d(i, f, k, s)))

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            h = tensor_in
            h = tf.expand_dims(tf.expand_dims(tf.expand_dims(h, 1), 1), 1)

            # Shared network
            with tf.variable_scope('shared'):
                h = tconv_layer(h, 512, (4, 1, 1), (4, 1, 1))        # 4, 1, 1
                h = tconv_layer(h, 256, (1, 4, 3), (1, 4, 3))        # 4, 4, 3
                h = tconv_layer(h, 128, (1, 4, 3), (1, 4, 2))        # 4, 16, 7

            # Pitch-time private network
            with tf.variable_scope('pitch_time_private'):
                s1 = [tconv_layer(h, 32, (1, 1, 12), (1, 1, 12))     # 4, 16, 84
                      for _ in range(self.n_tracks)]
                s1 = [tconv_layer(s1[i], 16, (1, 3, 1), (1, 3, 1))   # 4, 48, 84
                      for i in range(self.n_tracks)]

            # Time-pitch private network
            with tf.variable_scope('time_pitch_private'):
                s2 = [tconv_layer(h, 32, (1, 3, 1), (1, 3, 1))       # 4, 48, 7
                      for _ in range(self.n_tracks)]
                s2 = [tconv_layer(s2[i], 16, (1, 1, 12), (1, 1, 12)) # 4, 48, 84
                      for i in range(self.n_tracks)]

            h = [tf.concat((s1[i], s2[i]), -1) for i in range(self.n_tracks)]

            # Merged private network
            with tf.variable_scope('merged_private'):
                h = [norm(tconv3d(h[i], 1, (1, 1, 1), (1, 1, 1)))    # 4, 48, 84
                     for i in range(self.n_tracks)]
                h = tf.concat(h, -1)

        return tanh(h)
