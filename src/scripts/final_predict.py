from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib
import os
import pickle
import json
from sklearn.preprocessing import MinMaxScaler

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

logistic_regression=joblib.load(LOGISTIC_MODEL)
linear_svm_model=joblib.load(LINEAR_SVM_MODEL)
rbf_svm_model=joblib.load(RBF_SVM_MODEL)
linear_svm_model_mem=joblib.load(LINEAR_SVM_MODEL_MEM)
rbf_svm_model_mem=joblib.load(RBF_SVM_MODEL_MEM)
neural_net_model=joblib.load(NN_MODEL)

def get_frames(INPUT_FOLDER):
	# INPUT_FOLDER="../../data/tdata/out_BodyPullUps/"
	INPUT_FRAMES=os.listdir(INPUT_FOLDER)
	# print(len(INPUT_FRAMES))
	features_input=[]
	for frame in INPUT_FRAMES:
		# print(frame)
		with open(INPUT_FOLDER + frame) as f:
			data=json.load(f)
			if(len(data['people'])>0):
				if(len(data['people'][0]['pose_keypoints_2d'])>0):
					output=[]
					vector=data['people'][0]['pose_keypoints_2d']
					output_x=[]
					output_y=[]
					output_conf=[]
					for i in range(len(vector)):
						if(i%3==0):
							output_x.append(vector[i])
						elif((i-1)%3==0):
							output_y.append(vector[i])
						else:
							output_conf.append(vector[i])
					# print(len(vector))
					vector=np.asarray(vector)
					output_x=np.asarray(output_x)
					output_y=np.asarray(output_y)
					output_conf=np.asarray(output_conf)
					scaler=MinMaxScaler()
					output_x=scaler.fit_transform(output_x.reshape(-1, 1))
					output_y=scaler.fit_transform(output_y.reshape(-1, 1))
					output_conf=scaler.fit_transform(output_conf.reshape(-1, 1))
					# print("Shapes:",output_x.shape, output_y.shape, output_conf.shape)
					
					pointer=0
					for i in range(len(vector)):
						if(i%3==0):
							output.append(output_x[pointer])
						elif((i-1)%3==0):
							output.append(output_y[pointer])
						else:
							output.append(output_conf[pointer])
							pointer+=1

					output_=[]

					for i in range(3, 45):
						output_.append(output[i])
		filtered_output=[]
		# print(len(output_))
		for i in range(len(output_)):
			if((i+1)%3!=0):
				filtered_output.append(output_[i])
		features_input.append(list(filtered_output))

	features_input=np.asarray(features_input)
	features_input=np.asarray(features_input).reshape((len(features_input), 28))
	return features_input


def predict(input_folder, model):
	data=get_frames(input_folder)
	if(model==rbf_svm_model_mem or model==linear_svm_model_mem):
		# print(np.asarray(data).shape)
		li=[]
		for i in range(0,len(data),30):
			l=[]
			if(i+30>len(data)):
				break
			for j in range(i,i+min(30,len(data)-i)):
				l.append(data[j])
			li.append(l)
		data=li
		# print(np.asarray(data).shape)
		data=np.asarray(data).reshape((len(data), 28*30))
		out=model.predict(data)
	else:
		out=model.predict(data)
	out_arr=[]
	for i in range(5):
		out_arr.append(0)
	out_arr=np.asarray(out_arr)

	print(out)

	for i in range(len(out)):
		out_arr[out[i]]+=1

	out_class=np.argmax(out_arr, axis=0)
	print('Prediction:', end=' ')
	if(out_class==0):
		print('Pushup')
	elif(out_class==1):
		print('Pullup')
	elif(out_class==2):
		print('Wall Pushup')
	elif(out_class==3):
		print('Jumping Jack')
	else:
		print('Weight Squat')
	# print(out_class)

predict('test/', logistic_regression)
predict('test/', linear_svm_model)
predict('test/', rbf_svm_model)
predict('test/', linear_svm_model_mem)
predict('test/', rbf_svm_model_mem)
predict('test/', neural_net_model)


