# extract the data
test_data = []
with open('Part1_testing_data.txt') as reader:
    for line in reader:
        test_data.append(eval(line))
time = [i+1 for i in range(len(test_data))]

## the least squares estimation is done by Stata
## from the summary statistics in Stata, we choose N = 5
## That is because AIC is acceptably small when N = 5
## Display the results:
from sklearn.metrics import mean_squared_error
def calc_prediction(t):
    y_hat = 7617.603 - 123.464*t + 1.844261*(t**2) - 0.0059349*(t**3) - 0.0000325*(t**4) + 1.50e-07*(t**5)
    return y_hat
predictions = []
for t in time:
    predictions.append(calc_prediction(t))
error = mean_squared_error(test_data, predictions)
import numpy as np
def AIC(k, MSE):
    return 2*k - 4*np.log(MSE)
print('Test AIC: %.3f' % AIC(5,error))

# plot results
from matplotlib import pyplot
pyplot.plot(test_data)
pyplot.plot(predictions, color='red')
pyplot.show()