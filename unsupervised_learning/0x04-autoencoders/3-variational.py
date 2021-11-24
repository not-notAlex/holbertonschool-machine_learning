#!/usr/bin/env python3
"""
module for task 3
"""

import tensorflow.keras as keras


def sampling(args):
    """
    sampling
    """
    z_mean, z_log_sigma = args
    batch = keras.backend.shape(z_mean)[0]
    dim = keras.backend.int_shape(z_mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon





def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates a variational autoencoder
    """
    def kl_reconstruction_loss(true, pred):
        """
        k1 reconstruction loss
        """
        reconstruction_loss = keras.losses.binary_crossentropy(
            model_input, outputs)
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma - keras.backend.square(
            z_mean) - keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1) * -0.5
        return keras.backend.mean(reconstruction_loss + kl_loss)


    model_input = keras.layers.Input(shape=(input_dims,))
    encoded = model_input
    for layer in hidden_layers:
        encoded = keras.layers.Dense(layer, activation='relu')(encoded)
    z_mean = keras.layers.Dense(latent_dims)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims)(encoded)
    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoder = keras.Model(model_input, [z, z_mean, z_log_sigma])
    decoded = keras.layers.Input(shape=(latent_dims,))
    input_d = decoded
    for layer in reversed(hidden_layers):
        decoded = keras.layers.Dense(layer,  activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.models.Model(input_d, decoded)
    outputs = decoder(encoder(model_input))
    auto = keras.models.Model(model_input, outputs)
    auto.compile(optimizer='adam', loss=kl_reconstruction_loss)
    return encoder, decoder, auto
