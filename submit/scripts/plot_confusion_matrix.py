import numpy as np
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib

FILE_PUSHUPS=open('pushups.dat','rb')
FILE_PULLUPS=open('pullups.dat', 'rb')
FILE_WALLPUSHUPS=open('wallpushups.dat', 'rb')
FILE_JUMPINGJACK=open('bodyjumpingjacks.dat', 'rb')
FILE_JUMPROPE=open('bodyjumprope.dat', 'rb')
# FILE_PARALLELBARS=open('bodyparallelbars.dat', 'rb')
# FILE_UNEVENBARS=open('bodyunevenbars.dat', 'rb')
FILE_WEIGHTSQUATS=open('bodyweightsquats.dat', 'rb')
FILE_BOXINGPUNCHINGBAG=open('bodyboxingpunchingbag.dat', 'rb')
# FILE_HULAHOOP=open('bodyhulahoop.dat', 'rb')

LINEAR_SVM_MODEL="linear_svm_model.pickle"
RBF_SVM_MODEL="rbf_svm_model.pickle"
LOGISTIC_MODEL="logistic_model.pickle"

input_pushups=pickle.load(FILE_PUSHUPS)
input_pullups=pickle.load(FILE_PULLUPS)
input_wallpushups=pickle.load(FILE_WALLPUSHUPS)
input_jumpingjack=pickle.load(FILE_JUMPINGJACK)
input_jumprope=pickle.load(FILE_JUMPROPE)
# input_parallelbars=pickle.load(FILE_PARALLELBARS)
# input_unevenbars=pickle.load(FILE_UNEVENBARS)
input_weightsquats=pickle.load(FILE_WEIGHTSQUATS)
input_boxingpunchingbag=pickle.load(FILE_BOXINGPUNCHINGBAG)
# input_hulahoop=pickle.load(FILE_HULAHOOP)

input_pushups=np.asarray(input_pushups)
input_pullups=np.asarray(input_pullups)
input_wallpushups=np.asarray(input_wallpushups)
input_jumpingjack=np.asarray(input_jumpingjack)
input_jumprope=np.asarray(input_jumprope)
# input_parallelbars=np.asarray(input_parallelbars)
# input_unevenbars=np.asarray(input_unevenbars)
input_weightsquats=np.asarray(input_weightsquats)
input_boxingpunchingbag=np.asarray(input_boxingpunchingbag)
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

for i in range(len(input_jumprope)):
	data_x.append(input_jumprope[i])
	data_y.append(4)

# for i in range(len(input_parallelbars)):
# 	data_x.append(input_parallelbars[i])
# 	data_y.append(5)

# for i in range(len(input_unevenbars)):
# 	data_x.append(input_unevenbars[i])
# 	data_y.append(6)

for i in range(len(input_weightsquats)):
	data_x.append(input_weightsquats[i])
	data_y.append(7)

for i in range(len(input_boxingpunchingbag)):
	data_x.append(input_boxingpunchingbag[i])
	data_y.append(8)

# for i in range(len(input_hulahoop)):
# 	data_x.append(input_hulahoop[i])
# 	data_y.append(9)

data_x=np.asarray(data_x).reshape((len(data_x), 28))

training_x, test_x, training_y, test_y=train_test_split(data_x, data_y, test_size=0.2, shuffle=True)

# logistic_model=LogisticRegression(solver='newton-cg', multi_class='multinomial')
# logistic_model.fit(training_x, training_y)
logistic_model=joblib.load(LOGISTIC_MODEL)
predicted_y=logistic_model.predict(test_x)
print("Accuracy with newton-cg and multinomial:", accuracy_score(test_y, predicted_y))
sn.heatmap(pd.DataFrame(confusion_matrix(test_y, predicted_y), range(7), range(7)), annot=True, fmt='g')
plt.title("Confusion Matrix for logistic with nm")
plt.savefig('../plots/confusion_matrix_logistic.png')

# svm_model=svm.SVC(kernel='linear', gamma='auto', C=1000)
# svm_model.fit(training_x, training_y)
svm_model=joblib.load(LINEAR_SVM_MODEL)
predicted_y=svm_model.predict(test_x)
print("Accuracy for linear with C=1000:", accuracy_score(test_y, predicted_y))
sn.heatmap(pd.DataFrame(confusion_matrix(test_y, predicted_y), range(7), range(7)), annot=True, fmt='g')
plt.title("Confusion Matrix for linear svm with C=1000")
plt.savefig('../plots/confusion_matrix_svm_linear.png')

# svm_model=svm.SVC(kernel='rbf', gamma='auto', C=1000)
# svm_model.fit(training_x, training_y)
svm_model=joblib.load(RBF_SVM_MODEL)
predicted_y=svm_model.predict(test_x)
print("Accuracy for rbf with C=1000:", accuracy_score(test_y, predicted_y))
sn.heatmap(pd.DataFrame(confusion_matrix(test_y, predicted_y), range(7), range(7)), annot=True, fmt='g')
plt.title("Confusion Matrix for rbf svm with C=1000")
plt.savefig('../plots/confusion_matrix_svm_rbf.png')