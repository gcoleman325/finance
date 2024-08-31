# adapted from https://www.youtube.com/watch?v=9Y3yaoi9rUQ
from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

# TO DO: find survivorship bias free source
def get_stocks_in_index_from_wikipedia(link_to_list, years):
    src = pd.read_html(link_to_list)[0]
    src['Symbol'] = src['Symbol'].str.replace('.','-')
    symbols = src['Symbol'].unique().tolist()

    end_date = dt.date.today().strftime('%Y-%m-%d')
    start_date = pd.to_datetime(end_date)-pd.DateOffset(365*years)

    df = yf.download(tickers=symbols,
                     start=start_date,
                     end=end_date)

    df = df.stack()
    df.index.names = ['date', 'ticker']
    df.columns = df.columns.str.lower()

    return df

def compute_atr(df):
    atr = pandas_ta.atr(high = df['high'],
                        low = df['low'],
                        close = df['close'],
                        length = 14)
    return atr.sub(atr.mean()).div(atr.std())

def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())

def add_technical_analysis(df):
    high = np.log(df['high'])
    low = np.log(df['low'])
    adj_close = np.log(df['adj close'])
    open = np.log(df['open'])
    df['garman_klass_vol'] = ((high-low)**2)/2-(1*np.log(2)-1)*((adj_close-open)**2)

    df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close = x, length = 20))

    df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
    df['bb_middle'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
    df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])

    
    df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

    df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)
    
    df['dollar_volume'] = (df['adj close']*df['volume'])/1e6

    return df

def get_most_liquid(df):
    last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume','volume','open',
                                                            'high', 'low','close']]
    data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   df.unstack()[last_cols].resample('M').last().stack('ticker')],
                  axis=1)).dropna()
    data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())
    data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))
    data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)
    return data

def calculate_returns(df):
    outlier_cutoff = 0.005
    lags = [1,2,3,6,9,12]
    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                       upper=x.quantile(1-outlier_cutoff)))
                                .add(1)
                                .pow(1/lag)
                                .sub(1))
        return df
    
def factor_betas(df):
    df = df.groupby(level=1, group_keys=False).apply(calculate_returns)
    factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench')[0]
    factor_data.index = factor_data.index.to_timestamp()
    factor_data = factor_data.resample('M').last().div(100)
    factor_data.index.name = 'date'
    factor_data = factor_data.join(df['return_1m']).sort_index()
    observations = factor_data.groupby(level=1).size()

    valid_stocks = observations[observations >= 10]
    factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]
    
    betas = (factor_data.groupby(level=1,
                            group_keys=False)
         .apply(lambda x: RollingOLS(endog=x['return_1m'], 
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(24, x.shape[0]),
                                     min_nobs=len(x.columns)+1)
         .fit(params_only=True)
         .params
         .drop('const', axis=1)))

    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    data = (df.join(betas.groupby('ticker').shift()))
    data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
    data = data.drop('adj close', axis=1)

    return data

def get_clusters(df):
    data = df.dropna()
    kmeans = KMeans(n_clusters=4, random_state=0)
    data['cluster'] = kmeans.fit_predict(data)
    return data


df = get_stocks_in_index_from_wikipedia('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 3)
df = add_technical_analysis(df)
df = calculate_returns(df)
df = factor_betas(df)
print(get_clusters(df))
