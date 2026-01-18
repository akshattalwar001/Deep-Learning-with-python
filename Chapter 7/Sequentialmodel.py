"""
Demonstrates:
- Building a Keras Sequential model with add()
- Why weights do not exist before build()
- How build(input_shape) creates weights
- How to name models and layers
- What the outputs look like
"""

from tensorflow import keras
from tensorflow.keras import layers

# -------------------------------------------------
# Create Sequential model incrementally
# -------------------------------------------------
model = keras.Sequential()

# Add layers
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

# At this point, the model has NO weights
# Accessing model.weights here would raise:
# ValueError: Weights for model sequential have not yet been created.

# -------------------------------------------------
# Build the model by defining input shape
# -------------------------------------------------
# Input: 3 features, batch size unspecified (None)
model.build(input_shape=(None, 3))

# Weights are now created
print("Model weights:")
print(model.weights)

# Expected output:
# [
#  <tf.Variable 'dense/kernel:0' shape=(3, 64) dtype=float32>,
#  <tf.Variable 'dense/bias:0' shape=(64,) dtype=float32>,
#  <tf.Variable 'dense_1/kernel:0' shape=(64, 10) dtype=float32>,
#  <tf.Variable 'dense_1/bias:0' shape=(10,) dtype=float32>
# ]

# -------------------------------------------------
# Model summary
# -------------------------------------------------
model.summary()

# Expected output:
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                Output Shape              Param #
# =================================================================
# dense (Dense)               (None, 64)                256
# dense_1 (Dense)             (None, 10)                650
# =================================================================
# Total params: 906
# Trainable params: 906
# Non-trainable params: 0
# _________________________________________________________________


# -------------------------------------------------
# Same model with explicit names
# -------------------------------------------------
named_model = keras.Sequential(name="my_example_model")

named_model.add(
    layers.Dense(64, activation="relu", name="my_first_layer")
)
named_model.add(
    layers.Dense(10, activation="softmax", name="my_last_layer")
)

named_model.build((None, 3))
named_model.summary()

# Expected output:
# Model: "my_example_model"
# _________________________________________________________________
# Layer (type)                Output Shape              Param #
# =================================================================
# my_first_layer (Dense)      (None, 64)                256
# my_last_layer (Dense)       (None, 10)                650
# =================================================================
# Total params: 906
# Trainable params: 906
# Non-trainable params: 0
# _________________________________________________________________
