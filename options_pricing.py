import pandas as pd
import numpy as np 
import yfinance as yf
import datetime
import math
import statistics
from scipy.stats import norm
import random

def options_chain(ticker):
    # adapted from https://medium.com/@txlian13/webscrapping-options-data-with-python-and-yfinance-e4deb0124613
    ticker = yf.Ticker(ticker)
    exps = ticker.options
    options = pd.DataFrame()
    for exp in exps:
        opt = ticker.option_chain(exp)
        opt = pd.concat([opt.calls, opt.puts])
        opt['expirationDate'] = exp
        options = pd.concat([options, opt])

    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days = 1)
    options['dte'] = (options['expirationDate'] - pd.to_datetime('today')).dt.days / 365
    
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
    
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])
    options = options.reset_index(drop=True)

    return options

def monte_carlo(ticker):
    # implemented from https://www.investopedia.com/terms/m/montecarlosimulation.asp
    options = options_chain(ticker)
    totDailyReturns = []
    for i in range(len(options['mark']) - 1):
        if options['mark'][i] > 0 and options['mark'][i+1] > 0:
            totDailyReturns.append(np.log(options['mark'][i+1] / options['mark'][i]))
    
    variance = statistics.variance(totDailyReturns)
    drift = statistics.fmean(totDailyReturns) * (variance/2)
    
    last_price = options['mark'].iloc[-1]

    simulations = []
    for i in range(10000):
        random_value = math.sqrt(variance) * norm.ppf(random.random())
        price = last_price * math.exp(drift + random_value)
        simulations.append(price)

    return statistics.fmean(simulations)

def black_scholes(ticker, t):
    # implemented from https://intrinio.com/blog/how-the-black-scholes-option-pricing-model-works-its-benefits
    current_price = yf.Ticker(ticker).info['currentPrice']
    options = options_chain(ticker)
    strike = options['strike'].iloc[-1]

    totDailyReturns = []
    for i in range(len(options['mark']) - 1):
        if options['mark'][i] > 0 and options['mark'][i+1] > 0:
            totDailyReturns.append(np.log(options['mark'][i+1] / options['mark'][i]))
    
    variance = statistics.variance(totDailyReturns)
    volatility = math.sqrt(variance)

    # risk free rate adapted from https://gist.github.com/ranaroussi/72d0e92bbe31d1514baccf00175049e4
    risk_free_rate = yf.download("^IRX")['Adj Close'].iloc[-1] / 100
    r = (1 + risk_free_rate) ** (1/365) - 1

    d_1 = (np.log(current_price / strike) + (r + (variance / 2)) * t) / (volatility * math.sqrt(t))
    d_2 = d_1 - (volatility * math.sqrt(t))
    
    call = (current_price * norm.cdf(d_1)) - (strike * math.exp(-r * t) * norm.cdf(d_2))
    put = (strike * math.exp(-r * t) * norm.cdf(-d_2)) - (current_price * norm.cdf(-d_1))

    return call, put

call, put = black_scholes("GOOGL", 3)
print(call)
print(put)
