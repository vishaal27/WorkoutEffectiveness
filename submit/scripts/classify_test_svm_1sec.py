import numpy as np
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

FILE_PATH='../pickle_files/'

FILE_PUSHUPS=open(FILE_PATH+'pushups.dat','rb')
FILE_PULLUPS=open(FILE_PATH+'pullups.dat', 'rb')
FILE_WALLPUSHUPS=open(FILE_PATH+'wallpushups.dat', 'rb')
FILE_JUMPINGJACK=open(FILE_PATH+'bodyjumpingjacks.dat', 'rb')
# FILE_JUMPROPE=open('bodyjumprope.dat', 'rb')
# FILE_PARALLELBARS=open('bodyparallelbars.dat', 'rb')
# FILE_UNEVENBARS=open('bodyunevenbars.dat', 'rb')
FILE_WEIGHTSQUATS=open(FILE_PATH+'bodyweightsquats.dat', 'rb')
# FILE_BOXINGPUNCHINGBAG=open('bodyboxingpunchingbag.dat', 'rb')
# FILE_HULAHOOP=open('bodyhulahoop.dat', 'rb')

LINEAR_SVM_MODEL="../saved_models/linear_svm_model_mem.pickle"
RBF_SVM_MODEL="../saved_models/rbf_svm_model_mem.pickle"
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import h5py
from math import exp
import matplotlib.image as mpimg
import os
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.manifold import TSNE
from mlxtend.plotting import plot_decision_regions
# 

C_values=[0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 1000]
gamma_values=[0.0005, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.5, 0.1, 0.2, 0.3, 0.4, 1]

input_pushups=pickle.load(FILE_PUSHUPS)
li=[]
for i in range(0,len(input_pushups),30):
	l=[]
	if(i+30>len(input_pushups)):
		break
	for j in range(i,i+min(30,len(input_pushups)-i)):
		l.append(input_pushups[j])
	li.append(l)
input_pushups=li

input_pullups=pickle.load(FILE_PULLUPS)
lip=[]
for i in range(0,len(input_pullups),30):
	l=[]
	if(i+30>len(input_pullups)):
		break
	for j in range(i,i+min(30,len(input_pullups)-i)):
		l.append(input_pullups[j])
	lip.append(l)
input_pullups=lip


input_wallpushups=pickle.load(FILE_WALLPUSHUPS)
liwp=[]
for i in range(0,len(input_wallpushups),30):
	l=[]
	if(i+30>len(input_wallpushups)):
		break
	for j in range(i,i+min(30,len(input_wallpushups)-i)):
		l.append(input_wallpushups[j])
	liwp.append(l)
input_wallpushups=liwp


input_jumpingjack=pickle.load(FILE_JUMPINGJACK)
lijj=[]
for i in range(0,len(input_jumpingjack),30):
	l=[]
	if(i+30>len(input_jumpingjack)):
		break
	for j in range(i,i+min(30,len(input_jumpingjack)-i)):
		l.append(input_jumpingjack[j])
	lijj.append(l)
input_jumpingjack=lijj



input_weightsquats=pickle.load(FILE_WEIGHTSQUATS)
lijp=[]
for i in range(0,len(input_weightsquats),30):
	l=[]
	if(i+30>len(input_weightsquats)):
		break
	for j in range(i,i+min(30,len(input_weightsquats)-i)):
		l.append(input_weightsquats[j])
	lijp.append(l)
input_weightsquats=lijp

print(len(input_pushups),len(input_pullups),len(input_wallpushups),len(input_jumpingjack),len(input_weightsquats))



# input_parallelbars=pickle.load(FILE_PARALLELBARS)
# input_unevenbars=pickle.load(FILE_UNEVENBARS)
# input_weightsquats=pickle.load(FILE_WEIGHTSQUATS)
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

data_x=np.asarray(data_x).reshape((len(data_x), 28*30))
print(data_x.shape)

training_x, test_x, training_y, test_y=train_test_split(data_x, data_y, test_size=0.4, shuffle=True)


# svm_model=svm.SVC(kernel='linear', gamma='auto')
# grid_search_parameters={'C':C_values}
# grid_search=GridSearchCV(svm_model, grid_search_parameters, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(training_x, training_y)
# return_dict=grid_search.best_params_
# best_C=return_dict['C']

# Hardcode SVM

svm_model=svm.SVC(kernel='linear', gamma='auto', C=0.1)
svm_model.fit(training_x, training_y)
predicted_y=svm_model.predict(test_x)
print("Accuracy for linear with C=0.1:", accuracy_score(test_y, predicted_y))

print(svm_model.coef_.shape)

svm_model=svm.SVC(kernel='linear', gamma='auto', C=1)
svm_model.fit(training_x, training_y)
predicted_y=svm_model.predict(test_x)
print("Accuracy for linear with C=1:", accuracy_score(test_y, predicted_y))

svm_model=svm.SVC(kernel='linear', gamma='auto', C=10)
svm_model.fit(training_x, training_y)
predicted_y=svm_model.predict(test_x)
print("Accuracy for linear with C=10:", accuracy_score(test_y, predicted_y))

svm_model=svm.SVC(kernel='linear', gamma='auto', C=100)
svm_model.fit(training_x, training_y)
joblib.dump(svm_model, LINEAR_SVM_MODEL)
predicted_y=svm_model.predict(test_x)
print("Accuracy for linear with C=100:", accuracy_score(test_y, predicted_y))




# svm_model=svm.SVC(kernel='linear', gamma='auto', C=best_C)
# svm_model.fit(training_x, training_y)
# joblib.dump(svm_model, LINEAR_SVM_MODEL)
# predicted_y=svm_model.predict(test_x)
# print("Accuracy for linear with C="+str(best_C)+": "+str(accuracy_score(test_y, predicted_y)))

# svm_model=svm.SVC(kernel='rbf')
# grid_search_parameters={'C':C_values, 'gamma':gamma_values}
# grid_search=GridSearchCV(svm_model, grid_search_parameters, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(training_x, training_y)
# return_dict=grid_search.best_params_
# best_C=return_dict['C']
# best_gamma=return_dict['gamma']


# Hardcode svm

svm_model=svm.SVC(kernel='rbf', gamma='auto', C=0.5)
svm_model.fit(training_x, training_y)
predicted_y=svm_model.predict(test_x)
print("Accuracy for rbf with C=0.1:", accuracy_score(test_y, predicted_y))

svm_model=svm.SVC(kernel='rbf', gamma='auto', C=1)
svm_model.fit(training_x, training_y)
predicted_y=svm_model.predict(test_x)
print("Accuracy for rbf with C=1:", accuracy_score(test_y, predicted_y))

svm_model=svm.SVC(kernel='rbf', gamma='auto', C=10)
svm_model.fit(training_x, training_y)
joblib.dump(svm_model, RBF_SVM_MODEL)
predicted_y=svm_model.predict(test_x)
print("Accuracy for rbf with C=10:", accuracy_score(test_y, predicted_y))



# Hardcode svm

# svm_model=svm.SVC(kernel='rbf', gamma=best_gamma, C=best_C)
# svm_model.fit(training_x, training_y)
# joblib.dump(svm_model, RBF_SVM_MODEL)
# predicted_y=svm_model.predict(test_x)
# print("Accuracy for rbf with C="+str(best_C)+" and gamma="+str(best_gamma)+": "+str(accuracy_score(test_y, predicted_y)))