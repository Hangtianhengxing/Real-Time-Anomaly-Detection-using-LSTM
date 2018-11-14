from Dataprocessor import Dataloader
from keras.models import load_model
import os
import numpy as np
from matplotlib import pyplot
import lstm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings

model = load_model("lstm.h5")

seq_len = 300
data = Dataloader(seq_len,'concept drift.csv','TCP Connections Established/s')
x_train, y_train = data.get_train_data()
x_test,y_test = data.get_test_data()
#print (x_train.shape)
errordata = []

print ('\n[Model] Predicting point by point')

predicted_data = lstm.predict_point_by_point(model,x_test,y_test,x_train,y_train)

for i in range(len(predicted_data)):
	errordata.append(predicted_data[i]-y_test[i])
	
pyplot.plot(predicted_data,label = 'Predicted')
pyplot.plot(errordata, label = 'Error')
pyplot.plot(y_test,label = 'Actual')
pyplot.legend(['Predicted','Error','Actual'])
pyplot.show()

'''for i in range(len(predicted)):
	if (predicted[i]-y_test[i])>0.24:
		print ("Found anomaly at ",i+len(y_train))'''