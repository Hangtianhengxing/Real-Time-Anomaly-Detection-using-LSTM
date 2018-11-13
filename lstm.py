import os
import math
import numpy as np
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error as mse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings


def build_model(layers):
	model = Sequential()
	model.add(LSTM(input_dim = 1, output_dim= 50))
	#model.add(Dropout(0.5))
	#model.add(LSTM(256,return_sequences = False))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	#model.add(Activation("linear"))
	model.compile(loss="mse", optimizer="adam")
	#model.summary()
	return model
	
def predict_point_by_point(model,data,target):
#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
	print('[Model] Predicting Point-by-Point...')
	predicted = model.predict(data)
	'''predicted_data = []
	for i in range(len(data)):
		print (data[i])
		predicted = model.predict(data[i].shape[0],data[i].shape[1],1)
		print (mse(predicted,target[i]))
		predicted_data.append(predicted)'''
	predicted = np.reshape(predicted, (predicted.size))
	return predicted
	
