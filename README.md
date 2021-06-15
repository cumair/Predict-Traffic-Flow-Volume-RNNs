# Predict Traffic Flow Volume Using Time Series Analysis
## Neural Networks and Deep Learning

One of the most common applications of Time Series models is to predict future values. The goal of this project is to apply deep learning to do time series forecasting. In particular, create deep learning models to predict future traffic volume at a location in Minnesota, between Minneapolis and St Paul.

The specific goal will be to predict from a 6-hour input window, just the traffic volume for 2 hours past the end of the window.

To achieve this goal, TensorFlow (& Keras) are used to create and train Recurrent Neural Networks (RNNs) running various model configurations and hyperparameter values to empirically examine their effects in learning.

To search for an optimal model/hyperparameter setting for RNN, let's experiment with several different models and try exploring advanced features and architectures such as:

- Batch size
- Number of recurrent units
- Stacking recurrent layers
- Recurrent dropout
- Bidirectional RNNs
- Number of epochs

Finaly, we will submit our experimentations predictions for test dataset to Kaggle competition to compete with other students.

The data used in this project is metro interstate traffic volume data. It is publicly-available and acquired from UCI Machine Learning Repository. The traffic data was provided by MN Department of Transportation and the weather data was acquired using OpenWeatherMap.

The description of features present within this dataset are listed below which includes traffic volume, weather, and holiday information from 2012-2018.

- holiday: string (None or name of holiday) - Categorical
- temp: Average temp in kelvin - Numeric
- rain_1h: Amount in mm of rain that occurred in the hour - Numeric
- snow_1h: Amount in mm of snow that occurred in the hour - Numeric
- clouds: Percentage of cloud cover - Numeric
- weather_main: Short textual description of the current weather - Categorical
- weather_description: Longer textual description of the current weather - Categorical
- date_time: Hour of the data collected in local CST time - in M/D/Y H:m:s AM/PM format - DateTime
- traffic_volume: # of cars in the last hour - Numeric
