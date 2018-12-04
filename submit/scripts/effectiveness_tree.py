import numpy as np
import pickle
from sklearn import svm
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats

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

LINEAR_SVM_MODEL="../saved_models/linear_svm_model.pickle"
RBF_SVM_MODEL="../saved_models/rbf_svm_model.pickle"

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

means_pushups=[]
variances_pushups=[]

means_pullups=[]
variances_pullups=[]

means_wallpushups=[]
variances_wallpushups=[]

means_jumpingjack=[]
variances_jumpingjack=[]

means_weightsquats=[]
variances_weightsquats=[]

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

s1=0
s2=0
s3=0
s4=0
s5=0

arr1=[]
arr2=[]
arr3=[]
arr4=[]
arr5=[]

for i in range(len(input_pushups)-1):
	data_x.append(input_pushups[i])
	data_y.append(0)
	s1+=input_pushups[i][7]
	s2+=input_pushups[i][13]
	s3+=input_pushups[i][1]
	s4+=input_pushups[i][15]
	s5+=input_pushups[i][0]
	#pushups 7 13 1 15 0
	arr1.append(input_pushups[i][7])
	arr2.append(input_pushups[i][13])
	arr3.append(input_pushups[i][1])
	arr4.append(input_pushups[i][15])
	arr5.append(input_pushups[i][0])

means_pushups.append(s1/len(input_pushups))
means_pushups.append(s2/len(input_pushups))
means_pushups.append(s3/len(input_pushups))
means_pushups.append(s4/len(input_pushups))
means_pushups.append(s5/len(input_pushups))  

variances_pushups.append(np.var(arr1))
variances_pushups.append(np.var(arr2))
variances_pushups.append(np.var(arr3))
variances_pushups.append(np.var(arr4))
variances_pushups.append(np.var(arr5))

s1=0
s2=0
s3=0
s4=0
s5=0

arr1=[]
arr2=[]
arr3=[]
arr4=[]
arr5=[]

for i in range(len(input_pullups)-1):
	data_x.append(input_pullups[i])
	data_y.append(1)
	s1+=input_pullups[i][4]
	s2+=input_pullups[i][13]
	s3+=input_pullups[i][16]
	s4+=input_pullups[i][10]
	s5+=input_pullups[i][7]
	#pullups 4 13 16 10 7
	arr1.append(input_pullups[i][4])
	arr2.append(input_pullups[i][13])
	arr3.append(input_pullups[i][16])
	arr4.append(input_pullups[i][10])
	arr5.append(input_pullups[i][7])

means_pullups.append(s1/len(input_pullups))
means_pullups.append(s2/len(input_pullups))
means_pullups.append(s3/len(input_pullups))
means_pullups.append(s4/len(input_pullups))
means_pullups.append(s5/len(input_pullups)) 

variances_pullups.append(np.var(arr1))
variances_pullups.append(np.var(arr2))
variances_pullups.append(np.var(arr3))
variances_pullups.append(np.var(arr4))
variances_pullups.append(np.var(arr5))

s1=0
s2=0
s3=0
s4=0
s5=0

arr1=[]
arr2=[]
arr3=[]
arr4=[]
arr5=[]

for i in range(len(input_wallpushups)-1):
	data_x.append(input_wallpushups[i])
	data_y.append(2)
	s1+=input_wallpushups[i][6]
	s2+=input_wallpushups[i][20]
	s3+=input_wallpushups[i][3]
	s4+=input_wallpushups[i][18]
	s5+=input_wallpushups[i][10]
	#wallpushups 6 20 3 18 10 26
	arr1.append(input_wallpushups[i][6])
	arr2.append(input_wallpushups[i][20])
	arr3.append(input_wallpushups[i][3])
	arr4.append(input_wallpushups[i][18])
	arr5.append(input_wallpushups[i][10])

means_wallpushups.append(s1/len(input_wallpushups))
means_wallpushups.append(s2/len(input_wallpushups))
means_wallpushups.append(s3/len(input_wallpushups))
means_wallpushups.append(s4/len(input_wallpushups))
means_wallpushups.append(s5/len(input_wallpushups))

variances_wallpushups.append(np.var(arr1))
variances_wallpushups.append(np.var(arr2))
variances_wallpushups.append(np.var(arr3))
variances_wallpushups.append(np.var(arr4))
variances_wallpushups.append(np.var(arr5))

s1=0
s2=0
s3=0
s4=0
s5=0

arr1=[]
arr2=[]
arr3=[]
arr4=[]
arr5=[]

for i in range(len(input_jumpingjack)-1):
	data_x.append(input_jumpingjack[i])
	data_y.append(3)
	s1+=input_jumpingjack[i][12]
	s2+=input_jumpingjack[i][6]
	s3+=input_jumpingjack[i][4]
	s4+=input_jumpingjack[i][1]
	s5+=input_jumpingjack[i][5]
	#jumping jacks 12 6 4 1 5
	arr1.append(input_jumpingjack[i][12])
	arr2.append(input_jumpingjack[i][6])
	arr3.append(input_jumpingjack[i][4])
	arr4.append(input_jumpingjack[i][1])
	arr5.append(input_jumpingjack[i][5])

means_jumpingjack.append(s1/len(input_jumpingjack))
means_jumpingjack.append(s2/len(input_jumpingjack))
means_jumpingjack.append(s3/len(input_jumpingjack))
means_jumpingjack.append(s4/len(input_jumpingjack))
means_jumpingjack.append(s5/len(input_jumpingjack))

variances_jumpingjack.append(np.var(arr1))
variances_jumpingjack.append(np.var(arr2))
variances_jumpingjack.append(np.var(arr3))
variances_jumpingjack.append(np.var(arr4))
variances_jumpingjack.append(np.var(arr5))

# for i in range(len(input_jumprope)):
# 	data_x.append(input_jumprope[i])
# 	data_y.append(4)

# for i in range(len(input_parallelbars)):
# 	data_x.append(input_parallelbars[i])
# 	data_y.append(5)

# for i in range(len(input_unevenbars)):
# 	data_x.append(input_unevenbars[i])
# 	data_y.append(6)

s1=0
s2=0
s3=0
s4=0
s5=0

arr1=[]
arr2=[]
arr3=[]
arr4=[]
arr5=[]

for i in range(len(input_weightsquats)-1):
	data_x.append(input_weightsquats[i])
	data_y.append(4)
	s1+=input_weightsquats[i][25]
	s2+=input_weightsquats[i][22]
	s3+=input_weightsquats[i][24]
	s4+=input_weightsquats[i][27]
	s5+=input_weightsquats[i][7]
	#weight squats 25 22 24 27 7
	arr1.append(input_weightsquats[i][25])
	arr2.append(input_weightsquats[i][22])
	arr3.append(input_weightsquats[i][24])
	arr4.append(input_weightsquats[i][27])
	arr5.append(input_weightsquats[i][7])

means_weightsquats.append(s1/len(input_weightsquats))
means_weightsquats.append(s2/len(input_weightsquats))
means_weightsquats.append(s3/len(input_weightsquats))
means_weightsquats.append(s4/len(input_weightsquats))
means_weightsquats.append(s5/len(input_weightsquats))

variances_weightsquats.append(np.var(arr1))
variances_weightsquats.append(np.var(arr2))
variances_weightsquats.append(np.var(arr3))
variances_weightsquats.append(np.var(arr4))
variances_weightsquats.append(np.var(arr5))

# for i in range(len(input_boxingpunchingbag)):
# 	data_x.append(input_boxingpunchingbag[i])
# 	data_y.append(8)

# for i in range(len(input_hulahoop)):
# 	data_x.append(input_hulahoop[i])
# 	data_y.append(9)

print(len(input_pushups),len(input_pullups),len(input_wallpushups),len(input_jumpingjack),len(input_weightsquats))

print(np.asarray(data_x).shape)
data_x=np.asarray(data_x).reshape((len(data_x), 28))

training_x, test_x, training_y, test_y=train_test_split(data_x, data_y, test_size=0.2, shuffle=True)

freq=[]
for it in range(28):
	freq.append(0)

for it in range(3):

	decision_tree_model=DecisionTreeClassifier(max_depth=20)
	decision_tree_model.fit(training_x, training_y)
	predicted_y=decision_tree_model.predict(test_x)
	pr=decision_tree_model.predict(training_x)
	print("Accuracy:", accuracy_score(test_y, predicted_y))
	print("acc:", accuracy_score(training_y, pr))

	# print(decision_tree_model.tree_.max_depth)

	d={}
	for i in range(28):
		d[decision_tree_model.feature_importances_[i]]=i

	x=sorted(decision_tree_model.feature_importances_, reverse=True)

	for i in range(10):
		# print(x[i], d[x[i]])
		freq[d[x[i]]]+=1

for i in range(28):
	if(freq[i]!=0):
		print(i, freq[i])

print(np.asarray(means_pushups).shape, np.asarray(variances_pushups).shape)
print(means_pushups)


# plt.show()

print(test_x[0].shape)

ipp=np.asarray(input_pushups).reshape((len(input_pushups), 28))

# for i in range(5):
print(test_x[i])
pr=decision_tree_model.predict(ipp[:55])
print('class '+str(pr))

s1=0
s2=0
s3=0
s4=0
s5=0

arr1=[]
arr2=[]
arr3=[]
arr4=[]
arr5=[]

mean=[]
var=[]

for i in range(15):
	s1+=input_pullups[i][7]
	s2+=input_pushups[i][13]
	s3+=input_pushups[i][1]
	s4+=input_pushups[i][15]
	s5+=input_pushups[i][0]
	arr1.append(input_pushups[i][7])
	arr2.append(input_pushups[i][13])
	arr3.append(input_pushups[i][1])
	arr4.append(input_pushups[i][15])
	arr5.append(input_pushups[i][0])

mean.append(s1/5)
mean.append(s2/5)
mean.append(s3/5)
mean.append(s4/5)
mean.append(s5/5)
var.append(np.var(arr1))
var.append(np.var(arr2))
var.append(np.var(arr3))
var.append(np.var(arr4))
var.append(np.var(arr5))

plt.subplot(2,3,1)
val=np.random.normal(means_pushups[0], variances_pushups[0], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[0], var[0], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
# plt.show()

plt.subplot(2,3,2)
val=np.random.normal(means_pushups[1], variances_pushups[1], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[1], var[1], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
# plt.show()

plt.subplot(2,3,3)
val=np.random.normal(means_pushups[2], variances_pushups[2], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[2], var[2], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
# plt.show()

plt.subplot(2,3,4)
val=np.random.normal(means_pushups[3], variances_pushups[3], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[3], var[3], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
# plt.show()

plt.subplot(2,3,5)
val=np.random.normal(means_pushups[4], variances_pushups[4], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[4], var[4], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
plt.show()

st1=stats.entropy(np.random.normal(means_pushups[0], variances_pushups[0], 100), np.random.normal(mean[0], var[0], 100))
st2=stats.entropy(np.random.normal(means_pushups[1], variances_pushups[1], 100), np.random.normal(mean[1], var[1], 100))
st3=stats.entropy(np.random.normal(means_pushups[2], variances_pushups[2], 100), np.random.normal(mean[2], var[2], 100))
st4=stats.entropy(np.random.normal(means_pushups[3], variances_pushups[3], 100), np.random.normal(mean[3], var[3], 100))
st5=stats.entropy(np.random.normal(means_pushups[4], variances_pushups[4], 100), np.random.normal(mean[4], var[4], 100))

print(st1)
print(st2)
print(st3)
print(st4)
print(st5)

eff=0.2*(st1+st2+st3+st4+st5)
print('Effectiveness measure: '+str(eff))

s1=0
s2=0
s3=0
s4=0
s5=0

arr1=[]
arr2=[]
arr3=[]
arr4=[]
arr5=[]

mean=[]
var=[]

for i in range(15):
	s1+=input_pullups[i][4]
	s2+=input_pullups[i][13]
	s3+=input_pullups[i][16]
	s4+=input_pullups[i][10]
	s5+=input_pullups[i][7]
	arr1.append(input_pullups[i][4])
	arr2.append(input_pullups[i][13])
	arr3.append(input_pullups[i][16])
	arr4.append(input_pullups[i][10])
	arr5.append(input_pullups[i][7])

mean.append(s1/5)
mean.append(s2/5)
mean.append(s3/5)
mean.append(s4/5)
mean.append(s5/5)
var.append(np.var(arr1))
var.append(np.var(arr2))
var.append(np.var(arr3))
var.append(np.var(arr4))
var.append(np.var(arr5))


plt.subplot(2,3,1)
val=np.random.normal(means_pullups[0], variances_pullups[0], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[0], var[0], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
# plt.show()

plt.subplot(2,3,2)
val=np.random.normal(means_pullups[1], variances_pullups[1], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[1], var[1], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
# plt.show()

plt.subplot(2,3,3)
val=np.random.normal(means_pullups[2], variances_pullups[2], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[2], var[2], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
# plt.show()

plt.subplot(2,3,4)
val=np.random.normal(means_pullups[3], variances_pullups[3], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[3], var[3], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
# plt.show()

plt.subplot(2,3,5)
val=np.random.normal(means_pullups[4], variances_pullups[4], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[4], var[4], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
plt.show()

print()

st1=stats.entropy(np.random.normal(means_pullups[0], variances_pullups[0], 100), np.random.normal(mean[0], var[0], 100))
st2=stats.entropy(np.random.normal(means_pullups[1], variances_pullups[1], 100), np.random.normal(mean[1], var[1], 100))
st3=stats.entropy(np.random.normal(means_pullups[2], variances_pullups[2], 100), np.random.normal(mean[2], var[2], 100))
st4=stats.entropy(np.random.normal(means_pullups[3], variances_pullups[3], 100), np.random.normal(mean[3], var[3], 100))
st5=stats.entropy(np.random.normal(means_pullups[4], variances_pullups[4], 100), np.random.normal(mean[4], var[4], 100))

print(st1)
print(st2)
print(st3)
print(st4)
print(st5)

eff=0.2*(st1+st2+st3+st4+st5)
print('Effectiveness measure: '+str(eff))


s1=0
s2=0
s3=0
s4=0
s5=0

arr1=[]
arr2=[]
arr3=[]
arr4=[]
arr5=[]

mean=[]
var=[]

for i in range(15):
	s1+=input_wallpushups[i][4]
	s2+=input_wallpushups[i][13]
	s3+=input_wallpushups[i][16]
	s4+=input_wallpushups[i][10]
	s5+=input_wallpushups[i][7]
	arr1.append(input_wallpushups[i][4])
	arr2.append(input_wallpushups[i][13])
	arr3.append(input_wallpushups[i][16])
	arr4.append(input_wallpushups[i][10])
	arr5.append(input_wallpushups[i][7])

mean.append(s1/5)
mean.append(s2/5)
mean.append(s3/5)
mean.append(s4/5)
mean.append(s5/5)
var.append(np.var(arr1))
var.append(np.var(arr2))
var.append(np.var(arr3))
var.append(np.var(arr4))
var.append(np.var(arr5))


plt.subplot(2,3,1)
val=np.random.normal(means_wallpushups[0], variances_wallpushups[0], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[0], var[0], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
# plt.show()

plt.subplot(2,3,2)
val=np.random.normal(means_wallpushups[1], variances_wallpushups[1], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[1], var[1], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
# plt.show()

plt.subplot(2,3,3)
val=np.random.normal(means_wallpushups[2], variances_wallpushups[2], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[2], var[2], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
# plt.show()

plt.subplot(2,3,4)
val=np.random.normal(means_wallpushups[3], variances_wallpushups[3], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[3], var[3], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
# plt.show()

plt.subplot(2,3,5)
val=np.random.normal(means_wallpushups[4], variances_wallpushups[4], 10000)
plt.hist(val, 50)
val=np.random.normal(mean[4], var[4], 10000)
plt.hist(val, 50)
plt.xlim([0,3])
plt.show()

print()

st1=stats.entropy(np.random.normal(means_wallpushups[0], variances_wallpushups[0], 100), np.random.normal(mean[0], var[0], 100))
st2=stats.entropy(np.random.normal(means_wallpushups[1], variances_wallpushups[1], 100), np.random.normal(mean[1], var[1], 100))
st3=stats.entropy(np.random.normal(means_wallpushups[2], variances_wallpushups[2], 100), np.random.normal(mean[2], var[2], 100))
st4=stats.entropy(np.random.normal(means_wallpushups[3], variances_wallpushups[3], 100), np.random.normal(mean[3], var[3], 100))
st5=stats.entropy(np.random.normal(means_wallpushups[4], variances_wallpushups[4], 100), np.random.normal(mean[4], var[4], 100))

print(st1)
print(st2)
print(st3)
print(st4)
print(st5)

eff=0.2*(st1+st2+st3+st4+st5)
print('Effectiveness measure: '+str(eff))




#pushups 7 13 1 15 0  
#pullups 4 13 16 10 7
#wallpushups 6 20 3 18 10 26
#jumping jacks 12 6 4 1 5
#weight squats 25 22 24 27 7

# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8,  "MidHip"},
# {9,  "RHip"},
# {10, "RKnee"},
# {11, "RAnkle"},
# {12, "LHip"},
# {13, "LKnee"},
# {14, "LAnkle"},