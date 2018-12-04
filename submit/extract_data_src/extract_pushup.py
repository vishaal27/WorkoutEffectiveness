import numpy as np
import os
import json
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
import pickle

INPUT_FOLDER="../../data/tdata/out_Pushups/"
INPUT_FRAMES=os.listdir(INPUT_FOLDER)

features_input=[]
for frame in INPUT_FRAMES:
	print(frame)
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
				print("Shapes:",output_x.shape, output_y.shape, output_conf.shape)
				
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
	print(len(output_))
	for i in range(len(output_)):
		if((i+1)%3!=0):
			filtered_output.append(output_[i])
	features_input.append(list(filtered_output))

features_input=np.asarray(features_input)
output_file=open('pushups.dat','wb')
pickle.dump(features_input,output_file)
print(features_input.shape)

