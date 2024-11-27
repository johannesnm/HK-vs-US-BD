import yfinance as yf
import pandas as pd
import numpy as np

"""
Function to create a dataset based on the variables we want to include into the analysis.
Returns: A csv file as output: 'stock_data'. We can then load this file into the notebooks for further analysis!
"""

def fetch_and_save_stock_data(us_ticker='BABA', hk_ticker='9988.HK', start_date='2018-01-01', end_date='2024-11-5', period='5y', interval='1d', output_file='stock_data.csv'):
    # Fetch historical data
    us_data = yf.download(us_ticker, period=period, interval=interval)
    hk_data = yf.download(hk_ticker, period=period, interval=interval)

    # Additional features (VIX, SP500, HSI)
    vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close']
    hsi = yf.download('^HSI', start=start_date, end=end_date)['Close']

    # Combine the features into the DataFrame
    df = pd.DataFrame(index=us_data.index)
    df['US_Open'] = us_data['Open']
    df['US_Close'] = us_data['Close']
    df['HK_Close'] = hk_data['Close'] * 8 / 7.8
    df['HK_Open_next'] = hk_data['Open'].shift(-1) * 8 / 7.8
    df['VIX'] = vix
    df['SP500'] = sp500
    df['HSI'] = hsi

    # Adding Volume Data for both US and HK
    df['US_Volume'] = us_data['Volume']
    df['HK_Volume'] = hk_data['Volume']

    # Adding US 10-Year Treasury Yield and Forward Fill Missing Data
    treasury_yield = yf.download('^TNX', start=start_date, end=end_date)['Close']
    df['Treasury_Yield'] = treasury_yield

    # Adding Moving Averages (20-day) for both US and HK
    df['US_MA20'] = df['US_Close'].rolling(window=20).mean()
    df['HK_MA20'] = df['HK_Close'].rolling(window=20).mean()

    # Adding Relative Strength Index (RSI) for US Close Prices
    def calculate_rsi(data, window=14):
        delta = data.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df['RSI_US'] = calculate_rsi(df['US_Close'])

    # Adding Price Ratio (US to HK Close Prices)
    df['US_HK_Ratio'] = df['US_Close'] / df['HK_Close']

    # The values missing are likely due to holidays. Therefore, we fill in these values with the last known value they had.
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # Save DataFrame to CSV
    df.to_csv(output_file)

# Run the function and generate a dataset
fetch_and_save_stock_data()
