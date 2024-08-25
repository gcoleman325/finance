import pandas as pd
import numpy as np 
import yfinance as yf
import datetime
import math
import statistics
from scipy.stats import norm
import random

def options_chain(ticker):
    ticker = yf.Ticker(ticker)
    exps = ticker.options
    options = pd.DataFrame()
    for exp in exps:
        opt = ticker.option_chain(exp)
        opt = pd.concat([opt.calls, opt.puts])
        opt['expirationDate'] = exp
        options = pd.concat([options, opt])

    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + pd.Timedelta(days=1)
    options['dte'] = (options['expirationDate'] - pd.Timestamp.today()).dt.days / 365
    
    options['CALL'] = options['contractSymbol'].str[4:].apply(lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2
    
    options = options.drop(columns=['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])
    options = options.reset_index(drop=True)

    return options

def monte_carlo(ticker):
    options = options_chain(ticker)
    totDailyReturns = []
    for i in range(len(options['mark']) - 1):
        if options['mark'][i] > 0 and options['mark'][i+1] > 0:
            totDailyReturns.append(np.log(options['mark'][i+1] / options['mark'][i]))
    
    if len(totDailyReturns) == 0:
        raise ValueError("Not enough valid data for calculating returns.")
    
    variance = statistics.variance(totDailyReturns)
    drift = statistics.fmean(totDailyReturns) - (variance / 2)
    
    last_price = options['mark'].iloc[-1]

    simulations = []
    for i in range(10000):
        random_value = math.sqrt(variance) * norm.ppf(random.random())
        price = last_price * math.exp(drift + random_value)
        simulations.append(price)

    return statistics.fmean(simulations)

def black_scholes(ticker, t):
    ticker_data = yf.Ticker(ticker).info
    current_price = ticker_data.get('currentPrice', None)
    
    if current_price is None:
        raise ValueError("Current price data not available.")
    
    options = options_chain(ticker)
    strike = options['strike'].iloc[-1]

    totDailyReturns = []
    for i in range(len(options['mark']) - 1):
        if options['mark'][i] > 0 and options['mark'][i+1] > 0:
            totDailyReturns.append(np.log(options['mark'][i+1] / options['mark'][i]))
    
    if len(totDailyReturns) == 0:
        raise ValueError("Not enough valid data for calculating returns.")
    
    variance = statistics.variance(totDailyReturns)
    volatility = math.sqrt(variance) * math.sqrt(252) 

    risk_free_data = yf.download("^IRX")
    
    if risk_free_data.empty:
        raise ValueError("Risk-free rate data not available.")
    
    risk_free_rate = risk_free_data['Adj Close'].iloc[-1] / 100
    r = (1 + risk_free_rate) ** (1/252) - 1

    d_1 = (np.log(current_price / strike) + (r + (variance / 2)) * t) / (volatility * math.sqrt(t))
    d_2 = d_1 - (volatility * math.sqrt(t))
    
    call = (current_price * norm.cdf(d_1)) - (strike * math.exp(-r * t) * norm.cdf(d_2))
    put = (strike * math.exp(-r * t) * norm.cdf(-d_2)) - (current_price * norm.cdf(-d_1))

    return call, put

call, put = black_scholes("GOOGL", 3)
print(f"Call: {call}, Put: {put}")
print(monte_carlo("GOOGL"))
