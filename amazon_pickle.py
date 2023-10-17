import streamlit as st
import pickle
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the pre-trained SARIMA model.
with open('sarima_model.sav', 'rb') as f:
    model = pickle.load(f)

def fetch_stock_data(ticker=None, start_date=None, end_date=None):
    """Fetches and preprocesses stock data from yfinance.

    Args:
        ticker: The stock ticker symbol.
        start_date: The start date in datetime format.
        end_date: The end date in datetime format.

    Returns:
        A Pandas DataFrame containing the preprocessed stock data.
    """

    if ticker is None:
        ticker = 'AMZN'

    df = yf.download(ticker, start=start_date, end=end_date)

    # Apply the same preprocessing steps as the training data
    decomposed = seasonal_decompose(df['Adj Close'], model='additive')
    df['Adj Close'] = decomposed.resid

    return df

# Create the Streamlit app.
st.title('Stock Price Prediction App')

# Get the user-specified ticker symbol and time period.
ticker = st.text_input('Enter the stock ticker symbol:', value='AMZN')
start_date = st.date_input("Start date:", pd.to_datetime("2023-10-2"))
end_date = st.date_input('Enter the end date:', pd.to_datetime("2024-1-14"))

# Fetch and preprocess the stock data.
stock_data = fetch_stock_data(ticker=ticker, start_date=start_date, end_date=end_date)

# Check if the stock data DataFrame is empty.
if stock_data.empty:
    st.error('The stock data is empty. Please specify a different ticker symbol or time period.')
else:
    # Fit the SARIMA model to the stock data.
    try:
        results = model.fit(stock_data['Adj Close'])
    except:
        st.error("The SARIMA model could not be fit to the data. Please try a different ticker symbol or time period.")
    else:
        # Make a prediction.
        forecast = results.get_forecast(steps=90)
        prediction = forecast.predicted_mean

        # Display the original and predicted prices.
        st.write('Original Adjusted Close Price:', stock_data['Adj Close'])
        st.write('Predicted Adjusted Close Price:', prediction)
