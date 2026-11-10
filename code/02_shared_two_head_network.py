# Excerpt from the private training pipeline (train_capstone.py, model builder).
# One dense layer learns a representation used by both tasks. Each output head
# has its own loss during training. Not runnable on its own: `keras` and
# `input_dim` come from the surrounding script.

from tensorflow import keras

inputs = keras.Input(shape=(input_dim,), name="input")

# Shared representation used by both tasks; dropout reduces overfitting.
shared = keras.layers.Dense(128, activation="relu")(inputs)
shared = keras.layers.Dropout(0.5)(shared)

# Grade head: four class probabilities (D, C, B, A).
grade_output = keras.layers.Dense(4, activation="softmax", name="grade_output")(shared)
# Certificate head: a single probability.
cert_output = keras.layers.Dense(1, activation="sigmoid", name="cert_output")(shared)

model = keras.Model(inputs=inputs, outputs=[grade_output, cert_output])
