import utils
from shared_network import shared_network

import numpy as np

from turtle import distance
from keras.models import Input, Model
from keras.layers import Lambda


DATA_DIR = 'att_faces/'

(X_train, y_train), (X_test, y_test) = utils.load_data(DATA_DIR)
num_classes = len(np.unique(y_train))
# Train the model
training_pairs, training_labels = utils.create_pairs(X_train, y_train, num_classes=num_classes)

test_pairs, test_labels = utils.create_pairs(X_test, y_test, len(np.unique(y_test)))

input_shape = X_train.shape[1:]

# initiate model
shared_network = shared_network(input_shape)


input_network1 = Input(shape=input_shape)
input_network2 = Input(shape=input_shape)

# output 
output_network1 = shared_network(input_network1)
output_network2 = shared_network(input_network2)


distance = Lambda(utils.euclidian_distance, output_shape=(1,))([output_network1, output_network2])

model = Model(inputs=[input_network1, input_network2], outputs=distance)
print(model.summary())

model.compile(loss=utils.contrastive_loss, optimizer='adam')

model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_labels,
          batch_size=128,
          epochs=10)

# Save the model
model.save('siamese_nn.h5')