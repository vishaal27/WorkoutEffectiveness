import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

FILE_PATH='../pickle_files/'

FILE_PUSHUPS=open(FILE_PATH+'pushups.dat','rb')
FILE_PULLUPS=open(FILE_PATH+'pullups.dat', 'rb')
FILE_WALLPUSHUPS=open(FILE_PATH+'wallpushups.dat', 'rb')
FILE_JUMPINGJACK=open(FILE_PATH+'bodyjumpingjacks.dat', 'rb')
# FILE_JUMPROPE=open(FILE_PATH+'bodyjumprope.dat', 'rb')
# FILE_PARALLELBARS=open('bodyparallelbars.dat', 'rb')
# FILE_UNEVENBARS=open('bodyunevenbars.dat', 'rb')
FILE_WEIGHTSQUATS=open(FILE_PATH+'bodyweightsquats.dat', 'rb')
# FILE_BOXINGPUNCHINGBAG=open('bodyboxingpunchingbag.dat', 'rb')
# FILE_HULAHOOP=open('bodyhulahoop.dat', 'rb')

NN_MODEL="../saved_models/neural_net_model.pickle"

C_values=[0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 1000]
gamma_values=[0.0005, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.5, 0.1, 0.2, 0.3, 0.4, 1]

input_pushups=pickle.load(FILE_PUSHUPS)
input_pullups=pickle.load(FILE_PULLUPS)
input_wallpushups=pickle.load(FILE_WALLPUSHUPS)
input_jumpingjack=pickle.load(FILE_JUMPINGJACK)
# input_jumprope=pickle.load(FILE_JUMPROPE)
# input_parallelbars=pickle.load(FILE_PARALLELBARS)
# input_unevenbars=pickle.load(FILE_UNEVENBARS)
input_weightsquats=pickle.load(FILE_WEIGHTSQUATS)
# input_boxingpunchingbag=pickle.load(FILE_BOXINGPUNCHINGBAG)
# input_hulahoop=pickle.load(FILE_HULAHOOP)

input_pushups=np.asarray(input_pushups)
input_pullups=np.asarray(input_pullups)
input_wallpushups=np.asarray(input_wallpushups)
input_jumpingjack=np.asarray(input_jumpingjack)
# input_jumprope=np.asarray(input_jumprope)
# input_parallelbars=np.asarray(input_parallelbars)
# input_unevenbars=np.asarray(input_unevenbars)
input_weightsquats=np.asarray(input_weightsquats)
# input_boxingpunchingbag=np.asarray(input_boxingpunchingbag)
# input_hulahoop=np.asarray(input_hulahoop)

data_x=[]
data_y=[]

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

# for i in range(len(input_jumprope)):
# 	data_x.append(input_jumprope[i])
# 	data_y.append(4)

# for i in range(len(input_parallelbars)):
# 	data_x.append(input_parallelbars[i])
# 	data_y.append(5)

# for i in range(len(input_unevenbars)):
# 	data_x.append(input_unevenbars[i])
# 	data_y.append(6)

for i in range(len(input_weightsquats)):
	data_x.append(input_weightsquats[i])
	data_y.append(4)

# for i in range(len(input_boxingpunchingbag)):
# 	data_x.append(input_boxingpunchingbag[i])
# 	data_y.append(8)

# for i in range(len(input_hulahoop)):
# 	data_x.append(input_hulahoop[i])
# 	data_y.append(9)

data_x=np.asarray(data_x).reshape((len(data_x), 28))

training_x, test_x, training_y, test_y=train_test_split(data_x, data_y, test_size=0.2, shuffle=True)

neural_network_model=MLPClassifier(activation='logistic', hidden_layer_sizes=(100))
neural_network_model.fit(training_x, training_y)
joblib.dump(neural_network_model, NN_MODEL)
predicted_y=neural_network_model.predict(test_x)
print("Accuracy with logistic activation: "+str(accuracy_score(test_y, predicted_y)))

neural_network_model=MLPClassifier(activation='relu', hidden_layer_sizes=(100))
neural_network_model.fit(training_x, training_y)
joblib.dump(neural_network_model, NN_MODEL)
predicted_y=neural_network_model.predict(test_x)
print("Accuracy with logistic activation: "+str(accuracy_score(test_y, predicted_y)))