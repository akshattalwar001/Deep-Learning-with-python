from tensorflow.keras.datasets import mnist
(x_train , y_train), (x_test , y_test) = mnist.load_data()

print(x_train.shape) #3d array
y_train.shape #1d

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(512, activation="relu",input_shape=(784,)),
    Dense(10, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"]
              )

x_train = x_train.reshape((60000, 28*28))
x_train = x_train.astype("float32")/255
x_test = x_test.reshape((10000, 28*28))
x_test = x_test.astype("float32")/255

model.fit(x_train, y_train, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(x_test,y_test)
print(test_loss,test_acc)
#0.07090596854686737 0.9778000116348267