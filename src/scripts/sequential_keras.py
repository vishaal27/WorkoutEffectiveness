import numpy as np
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

FILE_PATH = '../pickle_files/'

FILE_PUSHUPS = open(FILE_PATH+'pushups.dat', 'rb')
FILE_PULLUPS = open(FILE_PATH+'pullups.dat', 'rb')
FILE_WALLPUSHUPS = open(FILE_PATH+'wallpushups.dat', 'rb')
FILE_JUMPINGJACK = open(FILE_PATH+'bodyjumpingjacks.dat', 'rb')
FILE_WEIGHTSQUATS = open(FILE_PATH+'bodyweightsquats.dat', 'rb')

NN_MODEL_RELU = "../saved_models/keras_sequential_relu.pickle"

input_pushups = pickle.load(FILE_PUSHUPS)
input_pullups = pickle.load(FILE_PULLUPS)
input_wallpushups = pickle.load(FILE_WALLPUSHUPS)
input_jumpingjack = pickle.load(FILE_JUMPINGJACK)
input_weightsquats = pickle.load(FILE_WEIGHTSQUATS)
input_pushups = np.asarray(input_pushups)
input_pullups = np.asarray(input_pullups)
input_wallpushups = np.asarray(input_wallpushups)
input_jumpingjack = np.asarray(input_jumpingjack)
input_weightsquats = np.asarray(input_weightsquats)

data_x = []
data_y = []

for i in range(len(input_pushups)):
	data_x.append(input_pushups[i])
	data_y.append(0)

for i in range(len(input_pullups)):
	data_x.append(input_pullups[i])
	data_y.append(1)

for i in range(len(input_wallpushups)):
	data_x.append(input_wallpushups[i])
	data_y.append(2)

for i in range(len(input_jumpingjack)):
	data_x.append(input_jumpingjack[i])
	data_y.append(3)

for i in range(len(input_weightsquats)):
	data_x.append(input_weightsquats[i])
	data_y.append(4)

data_x = np.asarray(data_x).reshape((len(data_x), 28))
# print(data_x.shape)
new_data_y = [] 
for x in data_y:
    new_x = np.zeros((10),dtype=int)
    new_x[x] = 1
    new_data_y.append(new_x)

data_y = np.asarray(new_data_y)


training_x, test_x, training_y, test_y = train_test_split(
	data_x, data_y, test_size=0.2, shuffle=True)

training_y = np.asarray(training_y)

nn_keras_model = Sequential()
nn_keras_model.add(Dense(100, activation='relu', input_dim=28))
nn_keras_model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

nn_keras_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

nn_keras_model.fit(training_x, training_y, epochs=200, batch_size=128)
nn_keras_score = nn_keras_model.evaluate(test_x, test_y, batch_size = 128)
joblib.dump(nn_keras_model, NN_MODEL_RELU)
print("Accuracy for MLP using Keras: "+str(nn_keras_score[1]))