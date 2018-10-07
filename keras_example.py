# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:50:08 2017

@author: dovah
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

np.random.seed(7)

dataset = np.loadtxt('pima-indians-diabetes.txt',delimiter=",")
x = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim = 8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x,y, epochs=150, batch_size=10)

scores = model.evaluate(x,y)

model.metrics_names[1]
scores[1]*100


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)

from matplotlib import pyplot as plt

#plt.imshow(x_train[2])
x_train = x_train.reshape(x_train.shape[0],1,28,28)
x_test = x_test.reshape(x_test.shape[0],1,28,28)

print(x_train.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(y_train.shape)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28),
                        dim_ordering='th'))
model.add(Convolution2D(32,3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
model.fit(x_train, y_train, batch_size=32, nb_epoch=10, verbose=1)
score=model.evaluate(x_test, y_test, verbose=0)

