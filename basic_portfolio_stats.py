import numpy as np
import pandas as pd
import yfinance as yf

def estimate_roi(ticker, initial, period):
    """
    :type ticker: str
    :type initial: int
    :type period: str
    :rtype: int
    """
    data = yf.Ticker(ticker).history(period=period)
    data = pd.DataFrame(data)

    col_idx = data.columns.get_loc('Close')
    start_price = data.iloc[0,col_idx]
    end_price = data.iloc[len(data)-1,col_idx]
    change = (end_price - start_price) / start_price

    return initial * change

print(estimate_roi('googl', 181, "1mo"))