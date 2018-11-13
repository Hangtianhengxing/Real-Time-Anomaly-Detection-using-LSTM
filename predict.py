from Dataprocessor import Dataloader
from keras.models import load_model
import os
import numpy as np
from matplotlib import pyplot
import lstm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings

model = load_model("lstm.h5")

seq_len = 200
data = Dataloader(seq_len,'concept drift.csv','TCP Connections Established/s')
x_test,y_test = data.get_test_data()
#print (x_test)
#print (y_test)
predicted_data = []
errordata = []

print ('\n[Model] Predicting point by point')

for i in range(len(x_test)):
	data = x_test[i].reshape(1,x_test[i].shape[0],1)
	target = np.array([y_test[i]])
	predicted = model.predict(data)
	predicted = np.reshape(predicted,predicted.size)
	#print (predicted)
	#print (target.shape)
	#model.fit(data,target,batch_size=1,nb_epoch=1)
	predicted_data.append(predicted)
	print (i)
	
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