#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers


class Seq2SeqModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, enc_units, dec_units, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.dec_units = dec_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.encoder = layers.Bidirectional(
            layers.LSTM(
                self.enc_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform'
            )
        )
        self.decoder = layers.LSTM(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.fc = layers.Dense(vocab_size)

    def call(self, enc_input, dec_input):
        enc_input = self.embedding(enc_input)
        enc_output, state_h_f, state_c_f, state_h_b, state_c_b = self.encoder(enc_input)
        state_h = layers.Concatenate()([state_h_f, state_h_b])
        state_c = layers.Concatenate()([state_c_f, state_c_b])
        encoder_state = [state_h, state_c]

        dec_input = self.embedding(dec_input)
        dec_output, _, _ = self.decoder(dec_input, initial_state=encoder_state)

        output = self.fc(dec_output)

        return output
