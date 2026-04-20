import pandas as pd
import yfinance as yf


def clean_stock_data_for_dashboard(ticker, start_date, end_date):
    """
    This function is called by the 'Run Analysis' button in app.py.
    It takes the values selected from the dropdowns and date picker.
    """
    # Fetch data based on the dashboard's dropdown values
    raw_df = yf.download(ticker, start=start_date,
                         end=end_date, auto_adjust=True)

    if raw_df.empty:
        return None

    # Flatten MultiIndex (prevents the KeyError seen earlier)
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)

    # Standardize columns for the 'Market Overview' tab
    df = raw_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Add the Technical Indicators seen in the UI
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # Add the 'Target' for the ML teammate's models
    df['Target'] = df['Close'].shift(-1)

    # Final Clean
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    # 1. Create the folder if it doesn't exist
    os.makedirs('data/ml_ready', exist_ok=True) 
    
    # 2. Define the filename based on the ticker
    file_path = f"data/ml_ready/{ticker}_training_data.csv"
    
    # 3. Save the dataframe to that path
    df.to_csv(file_path)
    
    print(f"ML-ready file created at: {file_path}")

    return df
