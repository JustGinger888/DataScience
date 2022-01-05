

from pickle import TRUE
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.ar_model import AutoRegResults
import numpy
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy
from math import sqrt

# create a difference transform of the dataset
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)

# Make a prediction give regression coefficients and lag obs
def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat

filename = 'data.xlsx'
df = pd.read_excel(filename)

filterEM = df['areaName']=='London'
df = df[filterEM]

df = df[['date', 'newCasesByPublishDate']]


# fit an AR model and save the whole model to file
from pandas import read_csv
from statsmodels.tsa.ar_model import AutoReg
import numpy

# create a difference transform of the dataset
def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return numpy.array(diff)




# split dataset
X = df['newCasesByPublishDate'].values
Y = df['date'].values

size = int(len(X) * 0.4)
train, testing = X[0:size], X[size:]
# train autoregression
model = AutoReg(train, lags=32)
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
history = [train[i] for i in range(len(train))]
predictions = list()
for t in range(len(testing)):
	yhat = predict(coef, history)
	obs = testing[t]
	predictions.append(yhat)
	history.append(obs)
rmse = sqrt(mean_squared_error(testing, predictions))
print('Test RMSE: %.3f' % rmse)
# plot

fill = len(X)-len(testing)
fillArr = []
for x in range(fill):
  fillArr.append(None)
predictions = fillArr + predictions



plt.plot(len(X))
plt.plot(len(testing))
plt.plot(predictions, color='red')


print(len(X))
print(len(test))
print(len(X)-len(test))

# # fit model
# model = AutoReg(X, lags=6)
# model_fit = model.fit()
# # save model to file
# model_fit.save('ar_model.pkl')
# # save the differenced dataset
# numpy.save('ar_data.npy', X)
# # save the last ob
# numpy.save('ar_obs.npy', [df.values[-1]])



# # load the AR model from file
# from statsmodels.tsa.ar_model import AutoRegResults
# import numpy
# loaded = AutoRegResults.load('ar_model.pkl')
# print(loaded.params)
# data = numpy.load('ar_data.npy', allow_pickle=TRUE)
# last_ob = numpy.load('ar_obs.npy', allow_pickle=TRUE)
# print(last_ob)

# load AR model from file and make a one-step prediction
# load model
# model = AutoRegResults.load('ar_model.pkl')
# data = numpy.load('ar_data.npy', allow_pickle=TRUE)
# last_ob = numpy.load('ar_obs.npy', allow_pickle=TRUE)
# # make prediction
# predictions = model.predict(start=len(data), end=len(data))
# # transform prediction
# yhat = predictions[0] + last_ob[0]
# print('Prediction: %f' , yhat)
