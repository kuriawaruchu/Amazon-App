import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from PIL import Image
import base64
#------------------------------------------------------------------------------#

# Page setup - the page title and layout
im = Image.open('icon.png')
st.set_page_config(page_title='The BullBear Oracle', page_icon=im,
    layout='wide', initial_sidebar_state="expanded")

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
st.title('**Welcome!**')
st.write('This app will help you view stock price predictions. Choose a stock symbol as listed on the [Yahoo! Finance website](https://finance.yahoo.com/lookup/) to get started.')
#------------------------------------------------------------------------------#

# Define a function to extract and preprocess the data
def load_and_preprocess_data(ticker):
    # Data extraction starts from 2018-01-01
    data = yf.download(ticker, start="2018-01-01")
    data.reset_index(inplace=True)
    columns_list = data.columns.to_list()
    
    # Allow the user to specify the date column and target column
    st.sidebar.subheader("Select the date column:")
    date_column = st.sidebar.selectbox("Date Column", columns_list)
    st.sidebar.subheader("Select the target column:")
    target_column = st.sidebar.selectbox("Target Column", columns_list)
    
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
        
        # Get the predicted values as a Pandas Series
        predicted_values = forecast.predicted_mean
        
        # Create a date range starting from the day after the last date in the data
        start_date = data.index[-1] + pd.DateOffset(days=1)
        date_range = pd.date_range(start=start_date, periods=len(predicted_values))

        # Create a DataFrame with 'Date' and 'Predicted' columns
        forecast_df = pd.DataFrame({'Date': date_range, 'Predicted': predicted_values})
    
    except Exception as e:
        st.write("Error: An error occurred during model fitting. Try different model parameters or data preprocessing.")
        return None

    return forecast_df
#------------------------------------------------------------------------------#
# Main Streamlit app
st.sidebar.header("Data Input")

# User input for the stock ticker
st.sidebar.subheader('Choose a Stock Symbol')
ticker = st.sidebar.text_input("Enter a stock ticker (e.g., AAPL):")
st.subheader(f"Your Predictions for {ticker} start here")

if ticker:
    data, date_column, target_column = load_and_preprocess_data(ticker)
    st.write("Data loaded and preprocessed successfully.")

    sarima_hyperparameters = DEFAULT_SARIMA_HYPERPARAMETERS

    st.sidebar.subheader("Select the end date for your predictions:")
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

    forecast_df = create_and_predict_sarima_model(data, date_column, target_column, DEFAULT_TRAIN_SIZE, sarima_hyperparameters, end_date)

    if forecast_df is not None:
        st.write("Model trained and predictions generated successfully.")
        forecast_df = forecast_df.set_index("Date")
        forecast_df.head()   
        
        if st.button('Save Your Predictions'):
            # Provide a download link for the predictions CSV
            csv_data = forecast_df.to_csv(index=False).encode()
            b64 = base64.b64encode(csv_data).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="stock_predictions.csv">Click here to download the predictions CSV file</a>'
            st.markdown(href, unsafe_allow_html=True)

        # Visualize predictions
        st.subheader(f'Predictions Graph for {ticker} until {end_date}')
        st.line_chart(forecast_df, y="Predicted")
        st.subheader(f"Predictions Table for {ticker} until {end_date}")
        st.table(forecast_df)
st.sidebar.subheader("About")
st.sidebar.write("The BullBears at Moringa School created this app. It uses pre-determined parameters of a SARIMA model to predict thestock prices.")
st.sidebar.write("We hope it helps!")