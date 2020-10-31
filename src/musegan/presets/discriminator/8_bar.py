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

"""This file defines the network architecture for the discriminator."""
import tensorflow as tf
from tensorflow.nn import relu, leaky_relu, tanh, sigmoid
from ..ops import dense, conv3d, get_normalization

NORMALIZATION = None # 'batch_norm', 'layer_norm', None
ACTIVATION = leaky_relu # relu, leaky_relu, tanh, sigmoid

class Discriminator:
    def __init__(self, n_tracks, beat_resolution=None, name='Discriminator'):
        self.n_tracks = n_tracks
        self.beat_resolution = beat_resolution
        self.name = name

    def __call__(self, tensor_in, condition=None, training=None):
        norm = get_normalization(NORMALIZATION, training)
        conv_layer = lambda i, f, k, s: ACTIVATION(norm(conv3d(i, f, k, s)))

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            h = tensor_in

            # Compute chroma feature
            n_beats = h.get_shape()[2] // self.beat_resolution
            reshaped = tf.reshape(
                tensor_in, (-1, h.get_shape()[1], n_beats, self.beat_resolution,
                            h.get_shape()[3], h.get_shape()[4]))
            summed = tf.reduce_sum(reshaped, 3)
            factor = int(h.get_shape()[3]) // 12
            remainder = int(h.get_shape()[3]) % 12
            reshaped = tf.reshape(
                summed[..., :(factor * 12), :],
                (-1, h.get_shape()[1], n_beats, factor, 12, h.get_shape()[4]))
            chroma = tf.reduce_sum(reshaped, 3)                      # 8, 4, 12
            if remainder:
                chroma += summed[..., -remainder:, :]

            # Compute onset/offset feature
            padded = tf.pad(tensor_in[:, :, :-1],
                            ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)))
            on_off_set = tf.reduce_sum(tensor_in - padded, 3, True)  # 8, 48, 1

            # Pitch-time private network
            with tf.variable_scope('pitch_time_private'):
                s1 = [conv_layer(h, 16, (1, 1, 12), (1, 1, 12))      # 8, 48, 7
                      for _ in range(self.n_tracks)]
                s1 = [conv_layer(s1[i], 32, (1, 3, 1), (1, 3, 1))    # 8, 16, 7
                      for i in range(self.n_tracks)]

            # Time-pitch private network
            with tf.variable_scope('time_pitch_private'):
                s2 = [conv_layer(h, 16, (1, 3, 1), (1, 3, 1))        # 8, 16, 84
                      for _ in range(self.n_tracks)]
                s2 = [conv_layer(s2[i], 32, (1, 1, 12), (1, 1, 12))  # 8, 16, 7
                      for i in range(self.n_tracks)]

            h = [tf.concat((s1[i], s2[i]), -1) for i in range(self.n_tracks)]

            # Merged private network
            with tf.variable_scope('merged_private'):
                h = [conv_layer(h[i], 64, (1, 1, 1), (1, 1, 1))      # 8, 16, 7
                     for i in range(self.n_tracks)]

            h = tf.concat(h, -1)

            # Shared network
            with tf.variable_scope('shared'):
                h = conv_layer(h, 128, (1, 4, 3), (1, 4, 2))         # 8, 4, 3
                h = conv_layer(h, 256, (1, 4, 3), (1, 4, 3))         # 8, 1, 1

            # Chroma stream
            with tf.variable_scope('chroma'):
                c = conv_layer(chroma, 32, (1, 1, 12), (1, 1, 12))   # 8, 4, 1
                c = conv_layer(c, 64, (1, 4, 1), (1, 4, 1))          # 8, 1, 1

            # Onset/offset stream
            with tf.variable_scope('on_off_set'):
                o = conv_layer(on_off_set, 16, (1, 3, 1), (1, 3, 1)) # 8, 16, 1
                o = conv_layer(o, 32, (1, 4, 1), (1, 4, 1))          # 8, 4, 1
                o = conv_layer(o, 64, (1, 4, 1), (1, 4, 1))          # 8, 1, 1

            h = tf.concat((h, c, o), -1)

            # Merge all streams
            with tf.variable_scope('merged'):
                h = conv_layer(h, 512, (6, 1, 1), (1, 1, 1))         # 3, 1, 1

            h = tf.reshape(h, (-1, h.get_shape()[-1]))
            h = dense(h, 1)

        return h