
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from offloading_gym.networks.tpto.multihead_attention import MultiHeadAttention



ACTOR_OUTPUT_DIM = 2
ACTOR_INNER_DIM = 256
CRITIC_OUTPUT_DIM = 1


class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, embed_dim, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu", kernel_initializer="he_normal"),
                tf.keras.layers.Dense(embed_dim, kernel_initializer="he_normal"),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization(
            gamma_initializer="he_normal", beta_initializer="he_normal", epsilon=1e-6
        )
        self.norm2 = tf.keras.layers.LayerNormalization(
            gamma_initializer="he_normal", beta_initializer="he_normal", epsilon=1e-6
        )
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training, mask=None):
        attn_output = self.att(
            query=inputs, value=inputs, key=inputs, attention_mask=mask
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.norm2(out1 + ffn_output)
# class Encoder(layers.Layer):
     
     
#     def __init__(self, num_heads, d_k, d_v, d_model, d_ff, num_enc_layers, dropout, **kwargs):
#         super().__init__(**kwargs)

#         self.d_model = d_model
           
#         self.input_sequence_1 = tf.keras.Sequential(
#             [
#                 layers.Dense(d_model, activation="tanh", kernel_initializer="he_normal"),
#                 layers.Dense(d_model, activation="tanh", kernel_initializer="he_normal"),
#             ]
#         )

#         self.input_sequence_2 = tf.keras.Sequential(
#             [
#                 layers.Dense(d_model, activation="tanh", kernel_initializer="he_normal"),
#                 layers.Dense(d_model, activation="tanh", kernel_initializer="he_normal"),
#             ]
#         )

#         self.dropout = layers.Dropout(dropout)

#         self.encoder_layer = [
#             TransformerBlock(num_heads, d_model, d_ff, dropout)
#             for _ in range(num_enc_layers)
#         ]

#         self.actor_layer = tf.keras.Sequential(
#             [
#                 layers.Dense(ACTOR_INNER_DIM, kernel_initializer="he_normal"),
#                 layers.BatchNormalization(),
#                 layers.Activation("tanh"),
#                 layers.Dense(ACTOR_INNER_DIM, kernel_initializer="he_normal"),
#                 layers.BatchNormalization(),
#                 layers.Activation("tanh"),
#                 layers.Dense(ACTOR_OUTPUT_DIM, activation="tanh", kernel_initializer="he_normal"),
#             ]
#         )

#         self.critic_layer = layers.Dense(CRITIC_OUTPUT_DIM, kernel_initializer="he_normal")

#     def call(self, inputs, training, mask=None):
#         input_1, input_2 = inputs

#         outputs_1 = self.input_sequence_1(input_1)
#         outputs_2 = self.input_sequence_2(input_2)
       
#         outputs_2 = tf.expand_dims(outputs_2, axis=2)  
#         outputs_2 = tf.tile(outputs_2, [1, 1, 57, 1])  
        

#         # Ensure correct dimension after concatenation
#         outputs = tf.concat([outputs_1, outputs_2], axis=-1)
#         outputs = tf.keras.layers.Dense(self.d_model, activation='tanh')(outputs) 
       
        

#         for block in self.encoder_layer:
#             outputs = block(outputs, training=training, mask=mask)

#         action = self.actor_layer(outputs)
#         value = self.critic_layer(outputs)
#         return action, value


class Encoder(layers.Layer):
    def __init__(self, num_heads, d_k, d_v, d_model, d_ff, num_enc_layers, dropout, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
           
        self.input_sequence_1 = tf.keras.Sequential(
            [
                layers.Dense(d_model, activation="tanh", kernel_initializer="he_normal"),
                layers.Dense(d_model, activation="tanh", kernel_initializer="he_normal"),
            ]
        )

        self.input_sequence_2 = tf.keras.Sequential(
            [
                layers.Dense(d_model, activation="tanh", kernel_initializer="he_normal"),
                layers.Dense(d_model, activation="tanh", kernel_initializer="he_normal"),
            ]
        )

        self.dropout = layers.Dropout(dropout)

        self.encoder_layer = [
            TransformerBlock(num_heads, d_model, d_ff, dropout)
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
                layers.Dense(ACTOR_OUTPUT_DIM, activation="tanh", kernel_initializer="he_normal"),
            ]
        )

        self.critic_layer = layers.Dense(CRITIC_OUTPUT_DIM, kernel_initializer="he_normal")

    def call(self, inputs, training, mask=None):
        input_1, input_2 = inputs

        outputs_1 = self.input_sequence_1(input_1)
        outputs_2 = self.input_sequence_2(input_2)
        
        # Expand and tile outputs_2 to match the sequence length of outputs_1
        batch_size = tf.shape(outputs_1)[0]
        sequence_length = tf.shape(outputs_1)[1]
        
        outputs_2 = tf.expand_dims(outputs_2, axis=1)  # Add sequence dimension
        outputs_2 = tf.tile(outputs_2, [1, sequence_length, 1])  # Tile to match sequence length

        # Concatenate outputs_1 and outputs_2
        outputs = tf.concat([outputs_1, outputs_2], axis=-1)
        
        # Pass through the Dense layer
        outputs = tf.keras.layers.Dense(self.d_model, activation='tanh')(outputs)

        for block in self.encoder_layer:
            outputs = block(outputs, training=training, mask=mask)

        action = self.actor_layer(outputs)
        value = self.critic_layer(outputs)
        return action, value


keras.utils.get_custom_objects()["Encoder"] = Encoder

