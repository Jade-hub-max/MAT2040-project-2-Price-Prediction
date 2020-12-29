# MAT2040-project-2-Price-Prediction
Contents
0	Introduction
1	Part One
	1.1	Autoregressive Model
	1.2	Fourier Series
	1.3	Taylor Formula
	1.4	Model Assessment
2	Part Two
2.1	Model Selection
2.2	Price Prediction
 
0.	Introduction
 
The Part1_training_data consists of n = 200 values which are prices of a product over time. We will analyze the dataset to determine the parameters of the three models. After that, Part1_testing_data consisting of n = 100 values will be used to test the validity of the three models. The standard for validity test is the Akaike information criterion (AIC).
Part 2 uses the Autoregressive Model to conduct both time series and panel data price prediction.
Data are analyzed using Python, RStudio and Stata. All code files in this report can be accessed using this GitHub link: https://github.com/Jade-hub-max/MAT2040-project-2-Price-Prediction. Please make sure that all files in the link are downloaded and stored in the same path before testing them.

1.	Part One
1.1	 Autoregressive Model
See Part1_AR.py
 
This figure shows values of the partial autocorrelation function (PACF). We choose n = 7 as this model’s lag. 
 
After the parameter is determined, we find the least square estimator of “an”, and apply the estimators to the testing data to calculate the AIC of this model. The AIC yields -30.645. 
1.2	 Fourier Series
See Part1_Fourier.Rmd
This part utilizes the “forest” library to get the parameters of the Fourier Series. Note that the time interval of this model is set as 100, which is the total length of this data, because the time series of the train data exhibits no periodicity.
 
After the parameter is determined, we find the least square estimator of “an” and “bn”, and apply the estimators to the testing data to calculate the AIC of this model. The AIC yields -35.216. 
1.3	 Taylor Formula
See Part1_Taylor.py & Stata_Taylor_n_selection.pdf
This part uses enumeration method in Stata to see which time intervals are appropriate. Based on the training data AIC, we choose the time interval n = 5.
 
 
After the parameter is determined, we find the least square estimator of “an” and “bn”, and apply the estimators to the testing data to calculate the AIC of this model (Alternatively, we can use NumPy to solve the least-squares problem through matrix operations such as projection and get the same result):
  
The AIC yields -48.003.
1.4	 Model Assessment
Since the AICs: -48.003 < -35.216 < -30.645, the Taylor expansion has the best fit when we use one product’s price to predict another’s price.

2.	Part Two
2.1	Model Selection
Part two uses Autoregressive Model to predict prices. That is because although the Taylor expansion has the least AIC, the Autoregressive Model is closely linked to price relationship. That caters to the requirement of relating one product’s price to other products’ price.
2.2	Price Prediction
See Part_2.py & Stata_Part2_interseries.pdf
This shows how to use AR to predict product_1's price, other products' prediction method uses the same method. To take into account both product_1’s past price and the other 4 products’ prices, we design a weighted average prediction here. 
We first use Stata to find the least-squares solution of the linear regression of product_1’s price on the other 4 products’ prices (Alternatively, we can use NumPy to solve the least-squares problem through matrix operations such as projection and get the same result):
 
Second, we conduct the autoregression on product_1 to get the numerical relationship between the current price and its own past price.
When both parts are done, we give a weighted average of them to output the final prediction:
 
After the prediction is determined, we apply the estimators to the testing data to calculate the AIC of this model. The AIC yields -35.789.

