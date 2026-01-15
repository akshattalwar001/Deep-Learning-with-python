from tensorflow.keras.datasets import imdb
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=10000) #only keep the top 10,000 most frequently occurring words in the training data

import numpy as np
import pandas as pd

x_train.shape #(25000,)

x_train[0] # a review = [1, 14, 22, 16, ... 178, 32]

word_index = imdb.get_word_index() #word_index is a dictionary mapping words to an integer index

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = " ".join(reverse_word_index.get(i - 3,"?") for i in x_train[0]) #Decodes the review. Note that the indices are offset by 3
                                                                        #because 0, 1, and 2 are reserved indices for “padding”,
                                                                      #“start of sequence,” and “unknown"

import numpy as np
def vectorize_sequences(sequences, dimension=10000):
 results = np.zeros((len(sequences), dimension))
 for i, sequence in enumerate(sequences):
  for j in sequence:
    results[i, j] = 1
 return results
x_tren = vectorize_sequences(x_train)
x_tst = vectorize_sequences(x_test)

y_train=np.asarray(y_train).astype('float32')
y_test=np.asarray(y_test).astype('float32')

from tensorflow import keras
from tensorflow.keras.layers import Dense
model = keras.Sequential([
 Dense(16, activation="relu"),
 Dense(16, activation="relu"),
 Dense(1, activation="sigmoid")
])

model.compile(optimizer="rmsprop",
 loss="binary_crossentropy",
 metrics=["accuracy"])

x_val = x_tren[:10000]
partial_x_train = x_tren[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model.fit(partial_x_train,
 partial_y_train,
 epochs=20,
 batch_size=512,
 validation_data=(x_val, y_val))

results = model.evaluate(x_tst, y_test)

#accuracy: 0.9987 - loss: 0.0155 - val_accuracy: 0.8696 - val_loss: 0.5663
''' | On training data | On validation data |
| ---------------- | ------------------ |
| Almost perfect   | Much worse         |
| Loss very low    | Loss very high     |
| 99.8% correct    | 87% correct        |

OVERFITTING


2 Dense layers

16 neurons each

Each neuron connects to all 10,000 input features.

weights = 10000 × 16 = 160,000

~160,000 parameters

For:

Only ~15,000 training reviews

it memorized all of it
 '''

model.fit(x_tren, y_train, epochs=4, batch_size=512)

results = model.evaluate(x_tst, y_test)

my_review = "this movie was absolutely fantastic with brilliant acting and a great story"

word_index = imdb.get_word_index()

def encode_review(text):
    tokens = text.lower().split()
    encoded = []
    for word in tokens:
        if word in word_index:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)   # unknown word
    return encoded
def vectorize(sequence, dimension=10000):
    x = np.zeros((1, dimension))
    for i in sequence:
        if i < dimension:
            x[0, i] = 1
    return x
encoded = encode_review(my_review)
vector = vectorize(encoded)

prediction = model.predict(vector)[0][0]

print("Review:", my_review)
print("Predicted probability of positive:", prediction)
print("Sentiment:", "Positive" if prediction > 0.5 else "Negative")
