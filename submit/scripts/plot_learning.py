from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib

def plot_learning_curve(classifier, title, data, labels, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), clr=1):
	print(classifier)
	plt.figure()
	plt.title(title)
	plt.xlabel('Training samples')
	plt.ylabel('Scores')
	plt.ylim(*ylim)
	train_sizes, train_scores, test_scores=learning_curve(classifier, data, labels, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
	train_scores_mean=np.mean(train_scores, axis=1)
	train_scores_std=np.std(train_scores, axis=1)
	test_scores_mean=np.mean(test_scores, axis=1)
	test_scores_std=np.std(test_scores, axis=1)
	plt.grid()
	plt.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std, alpha=0.1, color="red")
	plt.fill_between(train_sizes, test_scores_mean-test_scores_std, test_scores_mean+test_scores_std, alpha=0.1, color="blue")
	# plt.plot(train_sizes, train_scores_mean, 'o-', color="red", label="Training Score")
	if(clr==1):
		plt.plot(train_sizes, test_scores_mean, 'o-', color="blue", label="Test Score")
	elif(clr==2):
		plt.plot(train_sizes, test_scores_mean, 'o-', color="cyan", label="Test Score")
	elif(clr==3):
		plt.plot(train_sizes, test_scores_mean, 'o-', color="red", label="Test Score")
	elif(clr==4):
		plt.plot(train_sizes, test_scores_mean, 'o-', color="green", label="Test Score")
	elif(clr==5):
		plt.plot(train_sizes, test_scores_mean, 'o-', color="black", label="Test Score")
	else:
		plt.plot(train_sizes, test_scores_mean, 'o-', color="yellow", label="Test Score")
	# plt.legend(loc="best")
	# plt.show()



LINEAR_SVM_MODEL = "../saved_models/linear_svm_model.pickle"
RBF_SVM_MODEL = "../saved_models/rbf_svm_model.pickle"
LOGISTIC_MODEL = "../saved_models/logistic_model.pickle"
LINEAR_SVM_MODEL_MEM="../saved_models/linear_svm_model_mem.pickle"
RBF_SVM_MODEL_MEM="../saved_models/rbf_svm_model_mem.pickle"
NN_MODEL="../saved_models/neural_net_model.pickle"

FILE_PATH = '../pickle_files/'

FILE_PUSHUPS = open(FILE_PATH+'pushups.dat', 'rb')
FILE_PULLUPS = open(FILE_PATH+'pullups.dat', 'rb')
FILE_WALLPUSHUPS = open(FILE_PATH+'wallpushups.dat', 'rb')
FILE_JUMPINGJACK = open(FILE_PATH+'bodyjumpingjacks.dat', 'rb')
FILE_WEIGHTSQUATS = open(FILE_PATH+'bodyweightsquats.dat', 'rb')

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

training_x, test_x, training_y, test_y = train_test_split(
 data_x, data_y, test_size=0.2, shuffle=True)

logistic_regression = joblib.load(LOGISTIC_MODEL)
# linear_svm_model=joblib.load(LINEAR_SVM_MODEL)
rbf_svm_model=joblib.load(RBF_SVM_MODEL)
linear_svm_model_mem=joblib.load(LINEAR_SVM_MODEL_MEM)
rbf_svm_model_mem=joblib.load(RBF_SVM_MODEL_MEM)
neural_net_model=joblib.load(NN_MODEL)

plot_learning_curve(logistic_regression, 'Learning Curve', test_x, test_y, ylim=(0.0, 1.02), cv=5, n_jobs=-1, clr=1)
plot_learning_curve(linear_svm_model_mem, 'Learning Curve', test_x, test_y, ylim=(0.0, 1.02), cv=5, n_jobs=-1, clr=2)
# plot_learning_curve(linear_svm_model, 'Learning Curve', test_x, test_y, ylim=(0.0, 1.02), cv=5, n_jobs=-1, 3)
plot_learning_curve(rbf_svm_model, 'Learning Curve', test_x, test_y, ylim=(0.0, 1.02), cv=5, n_jobs=-1, clr=4)
plot_learning_curve(rbf_svm_model_mem, 'Learning Curve', test_x, test_y, ylim=(0.0, 1.02), cv=5, n_jobs=-1, clr=5)
plot_learning_curve(neural_net_model, 'Learning Curve', test_x, test_y, ylim=(0.0, 1.02), cv=5, n_jobs=-1, clr=6)
plt.show()