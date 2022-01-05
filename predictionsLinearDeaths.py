#Linear Regression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def ger_regression_predictions(features, intercept, slope):
    predicted_values = features*slope + intercept
    return predicted_values

filename = 'data.xlsx'
df = pd.read_excel(filename)

areaName = 'South West'
filter = df['areaName']==areaName
df = df[filter]
df = df[['newDeaths28DaysByPublishDate', 'date']]
data = df[['newDeaths28DaysByPublishDate', 'date']]

data['month'] = df['date'].dt.month
data['day'] = df['date'].dt.day
data['date'] = (df['date'].dt.month-1) * 31 + df['date'].dt.day

size = int(len(X) * 0.8)
train= data[:(int(len(X) * 0.8))]
test = data[(int(len(X) * 0.2)):]
# train autoregression
model = LinearRegression()
trainX = np.array(train[['date']])
trainY = np.array(train[['newDeaths28DaysByPublishDate']])
model.fit(trainX, trainY)
coef = model.coef_
intercept = model.intercept_

print(coef, intercept)

f = plt.figure()
f.set_figwidth(20)
f.set_figheight(10)

plt.plot(train['date'],train['newDeaths28DaysByPublishDate'])

plt.plot(trainX, coef*trainX + intercept, '-r')

estimated_values = ger_regression_predictions(180, intercept[0], coef[0][0])
print('estimated values: ',estimated_values)

plt.title('Linear regression of confirmed deaths by publish date in '+ areaName)
plt.ylim(bottom=-15, top=150)
plt.xlabel('Days')
plt.ylabel('New Deaths By Publish Date')
plt.grid(axis='x')
plt.show()
