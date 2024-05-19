#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


ACTOR_OUTPUT_DIM = 2
ACTOR_INNER_DIM = 256
CRITIC_OUTPUT_DIM = 1


class TransformerBlock(layers.Layer):
    """ A class representing a Transformer block. """
    def __init__(self, num_heads, embed_dim, output_dim, ff_dim, rate=0.1, **kwargs):
        """
        Creates a Transformer block.

        Args:
            num_heads (int): The number of attention heads.
            embed_dim (int): The dimension of the embedding.
            output_dim (int): The dimension of the output.
            ff_dim (int): The dimension of the feed-forward network hidden layer.
            rate (float): The dropout rate (default: 0.1).
        """
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, value_dim=embed_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu", kernel_initializer="he_normal"),
                layers.Dense(output_dim, kernel_initializer="he_normal"),
            ]
        )
        self.norm1 = layers.LayerNormalization(
            gamma_initializer="he_normal", beta_initializer="he_normal", epsilon=1e-6
        )
        self.norm2 = layers.LayerNormalization(
            gamma_initializer="he_normal", beta_initializer="he_normal", epsilon=1e-6
        )
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training, mask=None):
        """
        Args:
            inputs: (tf.Tensor) the input data.
            training: (bool) indicates whether the method is being called during training.
            mask: (tf.Tensor) an optional tensor representing the attention mask.

        Returns:
            The output tensor.
        """
        attn_output = self.att(
            query=inputs, value=inputs, key=inputs, attention_mask=mask
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)


class Encoder(layers.Layer):
    """

    The Encoder class is a subclass of the layers.Layer class in tensorflow.keras.
    It is used to implement the encoder part of a transformer model.

    Attributes:
        - num_heads: (int) the number of attention heads in each transformer block.
        - key_dim: (int) dimensionality of the query and key vectors in the attention mechanism.
        - embed_dim: (int) the dimensionality of the input embeddings.
        - output_dim: (int) the dimensionality of the output embeddings.
        - ff_dim: (int) the dimensionality of the feed-forward layer output in each transformer block.
        - num_enc_layers: (int) the number of transformer blocks in the encoder.
        - dropout: (float) the dropout rate.

    Methods:

        - call(self, inputs, training, mask)
            Executes the forward pass of the Encoder.
            Args:
                - inputs: A tensor containing the input sequence.
                - training: A boolean representing whether the model is in training mode or not.
                - mask: A tensor representing the mask used in the attention mechanism.
            Returns:
                - action: A tensor representing the output action.
                - value: A tensor representing the output value.
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        embed_dim,
        output_dim,
        ff_dim,
        num_enc_layers=256,
        dropout=0.0,
        **kwargs
    ):
        """
            Constructor.

            Args:
                num_heads: (int) the number of attention heads in each transformer block.
                key_dim: (int) dimensionality of the query and key vectors in the attention mechanism.
                embed_dim: (int) the dimensionality of the input embeddings.
                output_dim: (int) the dimensionality of the output embeddings.
                ff_dim: (int) the dimensionality of the feed-forward layer output in each transformer block.
                num_enc_layers: (int) the number of transformer blocks in the encoder.
                dropout: (float) the dropout rate.
                **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model_dim = output_dim

        self.input_sequence = tf.keras.Sequential(
            [
                layers.Dense(
                    output_dim, activation="tanh", kernel_initializer="he_normal"
                ),
                layers.Dense(
                    output_dim, activation="tanh", kernel_initializer="he_normal"
                ),
            ]
        )

        self.dropout = layers.Dropout(dropout)

        self.encoder_layer = [
            TransformerBlock(num_heads, key_dim, embed_dim, ff_dim, dropout)
            for _ in range(num_enc_layers)
        ]

        self.actor_layer = tf.keras.Sequential(
            [
                layers.Dense(ACTOR_INNER_DIM, kernel_initializer="he_normal"),
                layers.BatchNormalization(),
                layers.Activation("tanh"),
                layers.Dense(ACTOR_INNER_DIM, kernel_initializer="he_normal"),
                layers.BatchNormalization(),
                layers.Activation("tanh"),
                layers.Dense(
                    ACTOR_OUTPUT_DIM, activation="tanh", kernel_initializer="he_normal"
                ),
            ]
        )

        self.critic_layer = layers.Dense(
            CRITIC_OUTPUT_DIM, kernel_initializer="he_normal"
        )

    def call(self, inputs, training, mask=None):
        """
        Args:
            inputs: (tf.Tensor) the input sequence.
            training: (bool) indicates whether the model is in training mode or evaluation mode.
            mask: (tf.Tensor) an optional tensor representing the mask for the input sequence.

        Returns:
            A tuple of two tensors: action and value.
        """
        outputs = self.input_sequence(inputs)
        for block in self.encoder_layer:
            outputs = block(outputs, training=training, mask=mask)
        action = self.actor_layer(outputs)
        value = self.critic_layer(outputs)
        return action, value


keras.utils.get_custom_objects()["Encoder"] = Encoder
