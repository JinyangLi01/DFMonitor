

## source

https://github.com/akshitvjain/product-sales-forecasting/blob/master/notebook-sales-forecasting.ipynb

Holt Winter's Triple Exponential Smoothing Model

What is Exponential Smoothing?
This is a very popular scheme to produce a smoothed Time Series. Whereas in Single Moving Averages the past observations are weighted equally, Exponential Smoothing assigns exponentially decreasing weights as the observation get older. In other words, recent observations are given relatively more weight in forecasting than the older observations.
In the case of moving averages, the weights assigned to the observations are the same and are equal to 1/N. In exponential smoothing, however, there are one or more smoothing parameters to be determined (or estimated) and these choices determine the weights assigned to the observations.

There are 3 kinds of smoothing techniques Single, Double and Triple Exponential Smoothing.

Single Exponential Smoothing is used when the time series does not have a trend line and a seasonality component.
Double Exponential Smoothing is used to include forecasting data with a trend, smoothing calculation includes one for the level, and one for the trend.
Triple Exponential smoothing is used when data has trend and seasonality. We include a third equation to take care of seasonality (sometimes called periodicity). The resulting set of equations is called the "Holt-Winters" (HW) method after the names of the inventors.
Since out data has both trend and seasonality components, we will apply Triple Exponential Smoothing.


## data
The dataset contanins historical sales records of 10 stores and 50 products, from the year 2013 through 2017.

In this github, they only use the data of store 1

## Experiment

I plan to use the model to predict for all 10 stores
Store will be a sensitive attribute


