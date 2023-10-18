import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
import datetime

st.title(' Rough Stock Price Prediction App')

st.sidebar.header('Data Input')

# Initialize the data variable
data = None

# Function to train SARIMA model and generate predictions
def generate_sarima_predictions(df, target_column, sarima_hyperparameters, start_date, end_date):
    # Set the chosen datetime column as the index
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)

    # Filter data by user-selected start and end dates
    mask = (df.index >= start_date) & (df.index <= end_date)
    df = df.loc[mask]

    # Split the data into X and y variables
    y = df[target_column]
    X = df.drop(target_column, axis=1)

    # Split the data into train and test sets
    train_size = 0.8
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    # Fit the SARIMA model
    sarima_model = SARIMAX(y_train, order=(sarima_hyperparameters["p"], sarima_hyperparameters["d"], sarima_hyperparameters["q"]),
                          seasonal_order=(sarima_hyperparameters["seasonal_p"], sarima_hyperparameters["seasonal_d"],
                                          sarima_hyperparameters["seasonal_q"], sarima_hyperparameters["s"]))
    sarima_results = sarima_model.fit()

    # Generate predictions
    predictions = sarima_results.predict(start=len(y_train), end=len(df) - 1, dynamic=False)

    return predictions

# Option 1: User chooses input method
input_choice = st.sidebar.radio("Choose Input Method:", ("Predictions CSV File", "Data to Train", "Stock Symbol"))

# Option 2: User uploads a predictions CSV file or data to train
if input_choice in ["Predictions CSV File", "Data to Train"]:
    st.sidebar.subheader("Upload or provide a link to the CSV file:")
    csv_file_option = st.sidebar.radio("Choose Input Method:", ("Upload CSV File", "Link to CSV File"))

    if csv_file_option == "Upload CSV File":
        csv_file = st.sidebar.file_uploader("Upload CSV file:", type=["csv"])
        if csv_file:
            data = pd.read_csv(csv_file)
    else:
        csv_link = st.sidebar.text_input("Paste a link to CSV file:")
        if csv_link:
            data = pd.read_csv(csv_link)

    st.subheader('Select Date Range')
    start_date = st.date_input("Start date:", datetime.date(2022, 1, 1))
    end_date = st.date_input("End date:", datetime.date(2023, 1, 1))
    
    if input_choice == "Predictions CSV File":
        st.subheader('Define Columns')
        datetime_col = st.selectbox("Select the datetime column:", data.columns)
        target_col = st.selectbox("Select the target column:", data.columns)
        
        predictions = data
        
        # Convert the chosen datetime column to datetime
        data[datetime_col] = pd.to_datetime(data[datetime_col])
        
         # Visualize predictions using a line chart
        st.subheader(f'Predictions graph for {target_col}')
        st.line_chart(predictions)

    if input_choice == "Data to Train":
        st.subheader('Define Columns')
        datetime_col = st.selectbox("Select the datetime column:", data.columns)
        target_col = st.selectbox("Select the target column:", data.columns)

        # Convert the chosen datetime column to datetime
        data[datetime_col] = pd.to_datetime(data[datetime_col])

        sarima_hyperparameters = {
            "p": 2,
            "d": 0,
            "q": 2,
            "seasonal_p": 2,
            "seasonal_d": 0,
            "seasonal_q": 2,
            "s": 5,
        }

        # Train and generate predictions
        predictions = generate_sarima_predictions(data, target_col, sarima_hyperparameters, start_date, end_date)

        # Visualize predictions using a line chart
        st.subheader(f'Predictions graph for {target_col}')
        st.line_chart(predictions)

# Option 3: User provides a stock symbol
if input_choice == "Stock Symbol":
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL):")
    if stock_symbol:
        try:
            stock_data = yf.download(stock_symbol, start='2018-01-01')
            stock_data.reset_index(inplace=True)

            # Filter stock data by user-selected date range
            mask = (stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)
            stock_data = stock_data.loc[mask]

            # Convert stock data to a DataFrame
            predictions_df = stock_data[['Date', 'Adj Close']]
            predictions_df.columns = ['Datetime', 'Predicted_Adj_Close']

            st.subheader('Define Columns')
            datetime_col = st.selectbox("Select the datetime column:", predictions_df.columns)
            target_col = st.selectbox("Select the target column:", predictions_df.columns)

            # Convert the chosen datetime column to datetime
            predictions_df[datetime_col] = pd.to_datetime(predictions_df[datetime_col])

            sarima_hyperparameters = {
                "p": 2,
                "d": 0,
                "q": 2,
                "seasonal_p": 2,
                "seasonal_d": 0,
                "seasonal_q": 2,
                "s": 5,
            }

            # Train and generate predictions
            predictions = generate_sarima_predictions(predictions_df, target_col, sarima_hyperparameters, start_date, end_date)

            # Visualize predictions using a line chart
            st.subheader(f'Predictions graph for {target_col}')
            st.line_chart(data.set_index(datetime_col)[target_col])

        except Exception as e:
            st.write("Error: Could not fetch stock data. Please check the stock symbol.")