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

"""This file defines common GAN losses."""
import tensorflow as tf

def get_adv_losses(discriminator_real_outputs, discriminator_fake_outputs,
                   kind):
    """Return the corresponding GAN losses for the generator and the
    discriminator."""
    if kind == 'classic':
        loss_fn = classic_gan_losses
    elif kind == 'nonsaturating':
        loss_fn = nonsaturating_gan_losses
    elif kind == 'wasserstein':
        loss_fn = wasserstein_gan_losses
    elif kind == 'hinge':
        loss_fn = hinge_gan_losses
    return loss_fn(discriminator_real_outputs, discriminator_fake_outputs)

def classic_gan_losses(discriminator_real_outputs, discriminator_fake_outputs):
    """Return the classic GAN losses for the generator and the discriminator.

    (Generator)      log(1 - sigmoid(D(G(z))))
    (Discriminator)  - log(sigmoid(D(x))) - log(1 - sigmoid(D(G(z))))
    """
    discriminator_loss_real = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_real_outputs), discriminator_real_outputs)
    discriminator_loss_fake = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_fake_outputs), discriminator_fake_outputs)
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    generator_loss = -discriminator_loss
    return generator_loss, discriminator_loss

def nonsaturating_gan_losses(discriminator_real_outputs,
                             discriminator_fake_outputs):
    """Return the non-saturating GAN losses for the generator and the
    discriminator.

    (Generator)      -log(sigmoid(D(G(z))))
    (Discriminator)  -log(sigmoid(D(x))) - log(1 - sigmoid(D(G(z))))
    """
    discriminator_loss_real = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_real_outputs), discriminator_real_outputs)
    discriminator_loss_fake = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_fake_outputs), discriminator_fake_outputs)
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    generator_loss = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_fake_outputs), discriminator_fake_outputs)
    return generator_loss, discriminator_loss

def wasserstein_gan_losses(discriminator_real_outputs,
                           discriminator_fake_outputs):
    """Return the Wasserstein GAN losses for the generator and the
    discriminator.

    (Generator)      -D(G(z))
    (Discriminator)  D(G(z)) - D(x)
    """
    generator_loss = -tf.reduce_mean(discriminator_fake_outputs)
    discriminator_loss = -generator_loss - tf.reduce_mean(
        discriminator_real_outputs)
    return generator_loss, discriminator_loss

def hinge_gan_losses(discriminator_real_outputs, discriminator_fake_outputs):
    """Return the Hinge GAN losses for the generator and the discriminator.

    (Generator)      -D(G(z))
    (Discriminator)  max(0, 1 - D(x)) + max(0, 1 + D(G(z)))
    """
    generator_loss = -tf.reduce_mean(discriminator_fake_outputs)
    discriminator_loss = (
        tf.reduce_mean(tf.nn.relu(1. - discriminator_real_outputs))
        +  tf.reduce_mean(tf.nn.relu(1. + discriminator_fake_outputs)))
    return generator_loss, discriminator_loss
