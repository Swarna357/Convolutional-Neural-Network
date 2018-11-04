from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input 
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Sequential
import numpy as np

batch_size = 128
num_classes = 10
epochs = 10

#img_rows, img_cols = 28, 28
visible = Input(shape=(28,28,1))

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
input_shape = (28,28,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


conv1 = Conv2D(16, kernel_size=5, activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
flat1 = Flatten()(pool1)

conv2 = Conv2D(32, kernel_size=5, activation='relu')(visible)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat2 = Flatten()(pool2)

merge=concatenate([flat1,flat2])

hidden1 = Dense(10, activation='relu')(merge)

output = Dense(10, activation='sigmoid')(hidden1)
model = Model(inputs=visible, outputs=output)


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Train accuracy:', score[1])

