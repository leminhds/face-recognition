from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Lambda

import utils

import numpy as np

def shared_network(input_shape):
    model = Sequential(name='Shared_network')
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D())
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='sigmoid'))
    
    return model
