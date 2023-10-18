import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from PIL import Image
#------------------------------------------------------------------------------#

# Page setup - the page title and layout
im = Image.open('icon.png')
st.set_page_config(page_title='The BullBear Oracle', page_icon=im,
    layout='wide', initial_sidebar_state="expanded")
st.title('Rough 2')
# Default parameters for training and SARIMA model
DEFAULT_TRAIN_SIZE = 0.8
DEFAULT_SARIMA_HYPERPARAMETERS = {
    "p": 2,
    "d": 0,
    "q": 2,
    "seasonal_p": 2,
    "seasonal_d": 0,
    "seasonal_q": 2,
    "s": 5,
}

# Define a function to extract and preprocess the data
def load_and_preprocess_data(ticker):
    # Data extraction starts from 2018-01-01
    data = yf.download(ticker, start="2018-01-01")
    data.reset_index(inplace=True)
    columns_list = data.columns.to_list()
    
    # Allow the user to specify the date column and target column
    st.write("Select the date column:")
    date_column = st.selectbox("Date Column", columns_list)
    st.write("Select the target column:")
    target_column = st.selectbox("Target Column", columns_list)
    
    # Reset the index and set the selected date column as the index
    data.set_index(date_column, inplace=True)
    
    return data, date_column, target_column

# Define a function to create a SARIMA model and make predictions
def create_and_predict_sarima_model(data, date_column, target_column, train_size, sarima_hyperparameters, end_date):
    train_end = int(len(data) * train_size)
    train_data = data[:train_end]
    test_data = data[train_end:]

    try:
        # Create the SARIMA model
        sarima_model = SARIMAX(train_data[target_column], **sarima_hyperparameters)
        sarima_results = sarima_model.fit()

        # Use the SARIMA model to predict future dates
        forecast = sarima_results.get_forecast(steps=len(test_data) + len(pd.date_range(test_data.index[-1], end_date)))

        # Get predicted values and confidence intervals
        forecast_values = forecast.predicted_mean
        confidence_interval = forecast.conf_int()
        confidence_interval['Date'] = forecast_values.index
    except Exception as e:
        st.write("Error: An error occurred during model fitting . Try different model parameters or data preprocessing.")
        return None, None

    return forecast_values, confidence_interval

# Define a function to visualize predictions
def visualize_predictions(data, forecast_values, confidence_interval, target_column):
    fig = go.Figure()

    # Actual data
    fig.add_trace(go.Scatter(x=data.index, y=data[target_column], mode='lines', name='Actual', line=dict(color='green')))

    # Predicted values
    fig.add_trace(go.Scatter(x=forecast_values.index, y=forecast_values, mode='lines', name='SARIMA Forecast', line=dict(color='blue', shape='linear')))

    # Confidence interval
    fig.add_trace(go.Scatter(x=confidence_interval['Date'], y=confidence_interval['lower ' + target_column], fill=None, mode='lines', line=dict(color='red'), name='Lower CI'))
    fig.add_trace(go.Scatter(x=confidence_interval['Date'], y=confidence_interval['upper ' + target_column], fill='tonexty', mode='lines', line=dict(color='red'), name='Upper CI'))

    fig.update_xaxes(type='category')
    st.plotly_chart(fig)

# Main Streamlit app
st.title("Stock Price Prediction with SARIMA Model")

# User input for the stock ticker
ticker = st.text_input("Enter a stock ticker (e.g., AAPL):")

if ticker:
    data, date_column, target_column = load_and_preprocess_data(ticker)
    st.write("Data loaded and preprocessed successfully.")

    sarima_hyperparameters = DEFAULT_SARIMA_HYPERPARAMETERS

    st.write("Select the end date for predictions:")
    end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))

    forecast_values, confidence_interval = create_and_predict_sarima_model(data, date_column, target_column, DEFAULT_TRAIN_SIZE, sarima_hyperparameters, end_date)

    if forecast_values is not None:
        st.write("Model trained and predictions generated successfully.")

        # Visualize predictions
        st.write("Predictions:")
        visualize_predictions(data, forecast_values, confidence_interval, target_column)
