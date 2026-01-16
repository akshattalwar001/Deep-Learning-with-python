from tensorflow.keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
 num_words=10000)

import numpy as np
import pandas as pd

train_data.shape #80 20 split single column

len(train_data[0]) #array of 87 words index = 1 news

len(np.unique(train_labels)) #46 labels

word_index= reuters.get_word_index()

reversed_word_index=dict([(value,key) for (key,value) in word_index.items() ])
decoded_news=" ".join([reversed_word_index.get(i-3, "?") for i in train_data[0]])

decoded_news

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.
    return results

# 3. Create x_train
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 4. Now you can split
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

from tensorflow.keras.utils import to_categorical #one hot encoding like 6 = [0,0,0,0,0,1,0,0,...]
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

from tensorflow import keras
from keras.layers import Dense
model = keras.Sequential([
 Dense(64, activation="relu"),
 Dense(64, activation="relu"),
 Dense(46, activation="softmax")
])

# Before softmax:
# [2.3, 0.5, -1.1]

# After softmax:
# [0.78, 0.16, 0.06]
# Now the model can say:

# “78% sure it’s class 0”

model.compile(optimizer="rmsprop",
 loss="categorical_crossentropy",
 metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
    callbacks=[early_stop]
)

# final epoch 11 accuracy: 0.9372 - loss: 0.2982 - val_accuracy: 0.8170 - val_loss: 0.8909
model.evaluate(x_test, y_test) #accuracy: 0.8044 - loss: 0.9146
