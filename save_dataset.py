#Dummy dataset


import yfinance as yf
import pandas as pd
import os

# Ensure the data folder exists
os.makedirs("data", exist_ok=True)

# Download Apple stock data for 2018–2023
df = yf.download("AAPL", start="2018-01-01", end="2023-12-31", auto_adjust=True)

# Reset index so 'Date' becomes a column
df.reset_index(inplace=True)

# Save it as a CSV file with 'Date' column
df.to_csv("data/AAPL_2018_2023.csv", index=False)

print("Snapshot saved to data/AAPL_2018_2023.csv")
print(df.head())