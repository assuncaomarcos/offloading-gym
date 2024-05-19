import unittest
import numpy as np
import tensorflow as tf

from offloading_gym.networks import tpto


class TestEncoder(unittest.TestCase):

    def setUp(self):
        # We'll use a small shape for tractability
        self.num_heads = 8
        self.key_dim = 512
        self.ff_dim = 1024
        self.num_enc_layers = 3
        self.dropout_rate = 0.1

        # Create an instance of the Encoder layer
        self.encoder = tpto.Encoder(
            num_heads=self.num_heads, key_dim=self.key_dim,
            embed_dim=self.key_dim, output_dim=self.key_dim,
            ff_dim=self.ff_dim, num_enc_layers=self.num_enc_layers,
            dropout=self.dropout_rate
        )

    def test_call(self):
        # Create some mock data for the `call` method
        batch_size = 10
        seq_len = 16
        embedding_dim = self.key_dim

        inputs = tf.random.normal([batch_size, seq_len, embedding_dim])
        attention_mask = self.create_attention_mask(inputs)

        action, value = self.encoder(inputs, training=True, mask=attention_mask)

        self.assertIsInstance(action, tf.Tensor)
        self.assertIsInstance(value, tf.Tensor)

        self.assertEqual(action.shape, (batch_size, seq_len, 2))
        self.assertEqual(value.shape, (batch_size, seq_len, 1))

    @staticmethod
    def create_attention_mask(inputs):
        seq_len = inputs.shape[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        attention_mask = tf.random.uniform((seq_len, seq_len))
        attention_mask = tf.round(attention_mask)
        attention_mask = attention_mask * look_ahead_mask
        return attention_mask
