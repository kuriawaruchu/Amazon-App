import streamlit as st
import pickle
import pandas as pd
import yfinance as yf

# Load your SARIMA model
with open('sarima_model.pkl', 'rb') as file:
    sarima_model = pickle.load(file)

# Create a Streamlit app
st.title('The BullBear Oracle - Stock Price Prediction App')

# User chooses between using a CSV file or specifying a stock symbol
input_choice = st.sidebar.radio("Choose Input Method:", ("CSV File", "Stock Symbol"))

if input_choice == "CSV File":
    # User selects whether to upload a file or provide a link
    input_method = st.sidebar.radio("Choose Input Method:", ("Upload a CSV File", "Link to CSV"))

    if input_method == "Upload a CSV File":
        # Create input widget to upload a CSV file
        csv_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

        if csv_file:
            # Load data from the uploaded CSV file
            data = pd.read_csv(csv_file)

            # Create input widget to select the date column
            date_column = st.selectbox('Select Date Column', data.columns)
            data[date_column] = pd.to_datetime(data[date_column])  # Convert the selected date column to datetime

            # Create input widgets for user input
            prediction_date = st.date_input('Prediction Date')

            # Create a dropdown menu to select the column to predict
            column_to_predict = st.selectbox('Select Column to Predict', data.columns)

            # Perform prediction using your SARIMA model
            if st.button('Predict'):
                try:
                    # Filter data for the selected column to predict
                    selected_column_data = data[['Date', column_to_predict]]
                    selected_column_data.set_index('Date', inplace=True)

                    # Make sure your SARIMA model is capable of taking these inputs and returning predictions.
                    prediction = sarima_model.predict(selected_column_data, prediction_date)

                    # Display the prediction
                    st.write(f'Predicted {column_to_predict} on {prediction_date}: ${prediction:.2f}')
                except KeyError:
                    st.error('Invalid input or data not found. Please check your input and data.')
    else:
        # Create input widget to enter a link to the CSV file
        csv_link = st.sidebar.text_input("Enter the link to the CSV file")
        if csv_link:
            try:
                # Load data from the provided CSV file link
                data = pd.read_csv(csv_link)

                # Create input widget to select the date column
                date_column = st.selectbox('Select Date Column', data.columns)
                data[date_column] = pd.to_datetime(data[date_column])  # Convert the selected date column to datetime

                # Create input widgets for user input
                prediction_date = st.date_input('Prediction Date')

                # Create a dropdown menu to select the column to predict
                column_to_predict = st.selectbox('Select Column to Predict', data.columns)

                # Perform prediction using your SARIMA model
                if st.button('Predict'):
                    try:
                        # Filter data for the selected column to predict
                        selected_column_data = data[['Date', column_to_predict]]
                        selected_column_data.set_index('Date', inplace=True)

                        # Make sure your SARIMA model is capable of taking these inputs and returning predictions.
                        prediction = sarima_model.predict(selected_column_data, prediction_date)

                        # Display the prediction
                        st.write(f'Predicted {column_to_predict} on {prediction_date}: ${prediction:.2f}')
                    except KeyError:
                        st.error('Invalid input or data not found. Please check your input and data.')
            except Exception as e:
                st.error(f'An error occurred while loading data from the provided link: {str(e)}')

else:
    # User chooses to specify a stock symbol for prediction
    stock_symbol = st.sidebar.text_input('Enter Stock Symbol', 'AAPL')

    # Import stock data using yfinance
    try:
        stock_data = yf.download(stock_symbol, period="1d", interval="1d")
        stock_data.reset_index(inplace=True)

        # Set the 'Date' column as the index
        stock_data.set_index('Date', inplace=True)

        # Localize timezone if needed (example with 'America/New_York')
        # stock_data.index = stock_data.index.tz_localize('America/New_York')

        # Create input widgets for user input
        date_column = st.selectbox('Select Date Column', ['Date'])  # Assuming 'Date' is the date column
        prediction_date = st.date_input('Prediction Date')
        column_to_predict = st.selectbox('Select Column to Predict', stock_data.columns)

        # Perform prediction using your SARIMA model
        if st.button('Predict'):
            try:
                # Make sure your SARIMA model is capable of taking these inputs and returning predictions.
                prediction = sarima_model.predict(stock_data[[column_to_predict]], prediction_date)

                # Display the prediction
                st.write(f'Predicted {column_to_predict} for {stock_symbol} on {prediction_date}: ${prediction:.2f}')
            except Exception as e:
                st.error(f'An error occurred: {str(e)}')

    except Exception as e:
        st.error(f'An error occurred while fetching stock data: {str(e)}')
