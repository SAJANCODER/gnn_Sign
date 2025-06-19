import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np  # Often needed for custom layers even if not directly in the call method
import math  # Might be used in custom layer for calculations


# Custom GCN Layer
class CustomGCNConv(layers.Layer):
    def __init__(self, channels, activation=None, kernel_regularizer=None, **kwargs):
        super(CustomGCNConv, self).__init__(**kwargs)
        self.channels = channels
        self.activation = keras.activations.get(activation)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.dense = layers.Dense(channels, kernel_regularizer=self.kernel_regularizer, use_bias=False)

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="bias",
            shape=(self.channels,),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        X, A = inputs

        D_hat = tf.reduce_sum(A, axis=-1, keepdims=True)
        D_hat_inv_sqrt = tf.pow(D_hat + 1e-10, -0.5)
        D_hat_inv_sqrt = tf.where(tf.math.is_inf(D_hat_inv_sqrt), tf.zeros_like(D_hat_inv_sqrt), D_hat_inv_sqrt)

        A_norm = D_hat_inv_sqrt * A * tf.transpose(D_hat_inv_sqrt, perm=[0, 2, 1])

        X_transformed = self.dense(X)
        output = tf.matmul(A_norm, X_transformed)

        output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = super(CustomGCNConv, self).get_config()
        config.update({
            "channels": self.channels,
            "activation": keras.activations.serialize(self.activation),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


# Custom Global Sum Pooling Layer
class GlobalSumPool(layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalSumPool, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def get_config(self):
        return super(GlobalSumPool, self).get_config()

# You can optionally add a build_gnn_model function here too if you ever need to reconstruct the model,
# but for loading a saved model, just the layer definitions are critical. 