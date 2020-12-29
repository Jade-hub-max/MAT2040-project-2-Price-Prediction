# extract the data
train_data = []
with open('Part1_training_data.txt') as reader:
    for line in reader:
        train_data.append(eval(line))


# plot the time series of the training data
import matplotlib.pyplot as plt 
plt.plot([train_data.index(i) for i in train_data], train_data)
plt.title('Time series plot of training_data product price.')
plt.show()


## Autoregressive model
# Determine the lag by PACF
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf
train_data = np.array(train_data)
plot_pacf(train_data,lags=20)
plt.show()

## Get the OLS an
import pandas as pd
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
series_train = pd.read_csv('Part1_training_data.csv', header=0)
series_test = pd.read_csv('Part1_testing_data.csv', header=0)
train = series_train.values
test = series_test.values

# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)

# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
# AIC calculation
import numpy as np
def AIC(k, MSE):
    return 2*k - 4*np.log(MSE)
print('Test AIC: %.3f' % AIC(14,error))

# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()