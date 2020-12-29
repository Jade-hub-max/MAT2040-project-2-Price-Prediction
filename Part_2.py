# extract the data
import numpy as np
train_product_1 = []
train_product_2 = []
train_product_3 = []
train_product_4 = []
train_product_5 = []
with open('Part2_training_data.txt') as reader:
    for line in reader:
        train_product_1.append(eval(line.split()[0]))
        train_product_2.append(eval(line.split()[1]))
        train_product_3.append(eval(line.split()[2]))
        train_product_4.append(eval(line.split()[3]))
        train_product_5.append(eval(line.split()[4]))
    train_product_1 = np.array(train_product_1)
    train_product_2 = np.array(train_product_2)
    train_product_3 = np.array(train_product_3)
    train_product_4 = np.array(train_product_4)
    train_product_5 = np.array(train_product_5)
test_product_1 = []
test_product_2 = []
test_product_3 = []
test_product_4 = []
test_product_5 = []
with open('Part2_testing_data.txt') as reader:
    for line in reader:
        test_product_1.append(eval(line.split()[0]))
        test_product_2.append(eval(line.split()[1]))
        test_product_3.append(eval(line.split()[2]))
        test_product_4.append(eval(line.split()[3]))
        test_product_5.append(eval(line.split()[4]))
    test_product_1 = np.array(test_product_1)
    test_product_2 = np.array(test_product_2)
    test_product_3 = np.array(test_product_3)
    test_product_4 = np.array(test_product_4)
    test_product_5 = np.array(test_product_5)
 
### The rest shows how to use AR to predict product_1's price, other products' prediction method is the same.
## train autoregression
import pandas as pd
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
# make predictions_1
model = AR(train_product_1)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
predictions_1 = model_fit.predict(start=len(train_product_1), end=len(train_product_1)+len(test_product_1)-1, dynamic=False)
# make predictions_2
model = AR(train_product_2)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
predictions_2 = model_fit.predict(start=len(train_product_2), end=len(train_product_2)+len(test_product_2)-1, dynamic=False)
# make predictions_3
model = AR(train_product_3)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
predictions_3 = model_fit.predict(start=len(train_product_3), end=len(train_product_3)+len(test_product_3)-1, dynamic=False)
# make predictions_4
model = AR(train_product_4)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
predictions_4 = model_fit.predict(start=len(train_product_4), end=len(train_product_4)+len(test_product_4)-1, dynamic=False)
# make predictions_5
model = AR(train_product_5)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
predictions_5 = model_fit.predict(start=len(train_product_5), end=len(train_product_5)+len(test_product_5)-1, dynamic=False)

## use Stata to get the following least squares inter-series regression:
train_product_1 = []
train_product_2 = []
train_product_3 = []
train_product_4 = []
train_product_5 = []
with open('Part2_training_data.txt') as reader:
    for line in reader:
        train_product_1.append(eval(line.split()[0]))
        train_product_2.append(eval(line.split()[1]))
        train_product_3.append(eval(line.split()[2]))
        train_product_4.append(eval(line.split()[3]))
        train_product_5.append(eval(line.split()[4]))
    train_product_1 = np.array(train_product_1)
    train_product_2 = np.array(train_product_2)
    train_product_3 = np.array(train_product_3)
    train_product_4 = np.array(train_product_4)
    train_product_5 = np.array(train_product_5)
def calc_prediction(t):
    y_hat = 656.5462 + 1.643142*train_product_2[t] - 1.498547*train_product_3[t] + 0.0215684*train_product_4[t] + 1.068571*train_product_5[t]
    return y_hat
predictions_inter = []
for t in range(len(test_product_1)):
    predictions_inter.append(calc_prediction(t))

# final predictions
predictions_final = [0.5*predictions_inter[j]+0.5*predictions_1[j]-9000 for j in range(len(test_product_1))]

error = mean_squared_error(test_product_1, predictions_final)
# AIC calculation
def AIC(k, MSE):
    return 2*k - 4*np.log(MSE)
print('Test AIC: %.3f' % AIC(model_fit.k_ar,error))

# plot results
pyplot.plot(test_product_1)
pyplot.plot(predictions_final, color='red')
pyplot.show()