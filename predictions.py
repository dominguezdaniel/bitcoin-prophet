###### Step 1: Install And Import Libraries
# Install libraries
!pip install yfinance prophet
# Data processing
import numpy as np
import pandas as pd
# Get time series data
import yfinance as yf
# Prophet model for time series forecast
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics
# Visualization
import plotly.graph_objs as go
###### Step 2: Get Bitcoin Price Data
# Download Bitcoin data
data = yf.download(tickers='BTC-USD', start='2018-01-01', end='2019-12-31', interval = '1d')
# Reset index and have date as a column
data.reset_index(inplace=True)
# Change date to datetime format
data['Date'] = pd.to_datetime(data['Date'])
# Take a look at the data
data.head()
# Declare a figure
fig = go.Figure()
# Candlestick chart
fig.add_trace(go.Candlestick(x=data.Date,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'], 
                name = 'Bitcoin Data'))
# Keep only date and close price
df = data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
# Rename date to ds and close price to y
df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
# Take a look at the data
df.head()
# Data information
df.info()
###### Step 3: Train Test Split
# Train test split
df_train = df[df['ds']<='2019-11-30']
df_test = df[df['ds']>'2019-11-30']
###### Step 4: Train Time Series Model Using Prophet
# Create the prophet model with confidence internal of 95%
m = Prophet(interval_width=0.95, n_changepoints=7)
# Fit the model using the training dataset
m.fit(df_train)
###### Step 5: Use Prophet Model To Make Prediction
# Create a future dataframe for prediction
future = m.make_future_dataframe(periods=31)
# Forecast the future dataframe values
forecast = m.predict(future)
# Check the forecasted values and upper/lower bound
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# Visualize the forecast
fig = m.plot(forecast)
ax = fig.gca()
ax.plot( df_test["ds"], df_test["y"], 'r.')
###### Step 6: Time Series Decomposition
# Visualize the components
m.plot_components(forecast);
###### Step 7: Identify Change Points
# Default change points
print(f'There are {len(m.changepoints)} change points. \nThe change points dates are \n{df.loc[df["ds"].isin(m.changepoints)]}')
# Change points to plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
###### Step 8: Cross Validation
# Cross validation
df_cv = cross_validation(m, initial='500 days', period='60 days', horizon = '30 days', parallel="processes")
df_cv.head()
###### Step 9: Prophet Model Performance Evaluation
# Model performance metrics
df_p = performance_metrics(df_cv)
df_p.head()
# Visualize the performance metrics
fig = plot_cross_validation_metric(df_cv, metric='mape')
