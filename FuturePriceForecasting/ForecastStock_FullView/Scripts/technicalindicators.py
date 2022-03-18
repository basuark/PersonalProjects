import pandas as pd
import numpy as np

def rsi(datadf, windowdays=14):
    '''
    Calculates and returns rsi values for given window size
    RSI=100-[100/(1+(Average gain)/(Average Loss))] , Default windowsize=14 days
    '''
    length = len(datadf)
    datadf['Up'] = 0
    datadf['Down'] = 0
    datadf['AverageGain'] =0
    datadf['AverageLoss'] =0
    datadf['RStrength']=0
    datadf['RSI']=0
    
    for i in range(1,length):
        if datadf['Close'][i] >=datadf['Close'][i-1] :
            datadf['Up'][i] = datadf['Close'][i] - datadf['Close'][i-1]
        else:
            datadf['Down'][i] = abs(datadf['Close'][i] - datadf['Close'][i-1])
    
    # Calculating average for 14 days RSI
    for i in range(windowdays,length):
        if i==windowdays:
            datadf['AverageGain'][i] = datadf['Up'][1:(i+1)].mean()
            datadf['AverageLoss'][i] = datadf['Down'][1:(i+1)].mean()
        else:
            datadf['AverageGain'][i] = ((windowdays-1)*datadf['AverageGain'][i-1]+datadf['Up'][i])/windowdays
            datadf['AverageLoss'][i] = ((windowdays-1)*datadf['AverageLoss'][i-1]+datadf['Down'][i])/windowdays
            
        datadf['RStrength'][i] = datadf['AverageGain'][i]/datadf['AverageLoss'][i]
        datadf['RSI'][i] = 100 - (100/(1+datadf['RStrength'][i]))
    
    datadf = datadf.drop(['Up','Down','AverageGain','AverageLoss','RStrength'],axis=1)
    
    return datadf


# Calculating Stochaststic RSI values
def stochastic_RSI(datadf, windowsize=14):
    '''
    This function computes stochastic RSI values using RSI values 
    StochasticRSI = (RSI - min (RSI))/(max(RSI) - min(RSI)) , default look back period 14 days
    '''
    length=len(datadf)
    datadf['minRSI'] =0
    datadf['maxRSI'] =0
    datadf['stockRSI'] =0
    for i in range(windowsize,length):
        datadf['minRSI'] = datadf['RSI'][i-windowsize : i+1 ].min()
        datadf['maxRSI'] = datadf['RSI'][i-windowsize : i+1 ].max()
    
    datadf['stockRSI'] = (datadf['RSI'] - datadf['minRSI'])/(datadf['maxRSI'] - datadf['minRSI'])
    
    datadf = datadf.drop(['minRSI','maxRSI'],axis=1)
    return datadf

# MACD Calculation
def macd(datadf,emawindowLow=12,windowHigh=26,strengthLkp=9):
    '''
    This function calculates MACD momentum and strength
    MACD = EMA(12days)- EMA(26 days) , Strength = MACD - EMA(MACD,9days)
    '''
    datadf['ema_low'] = datadf['Close'].ewm(span=emawindowLow,adjust=False, min_periods=emawindowLow).mean()
    datadf['ema_high'] = datadf['Close'].ewm(span=windowHigh,adjust=False, min_periods=windowHigh).mean()
    
    datadf['macd_strength'] = datadf['Close'].ewm(span=strengthLkp,adjust=False, min_periods=strengthLkp).mean()
    
    datadf['macd'] = datadf['ema_low'] - datadf['ema_high']
    datadf['macd_h'] = datadf['macd'] - datadf['macd_strength']
    
    datadf = datadf.drop(['ema_low','ema_high'],axis=1)
    
    return datadf
    
# Exponential Moving Average Calculation
def current_EMA(datadf,ma_days=21):
    '''
    Calculates Exponential Moving Average for ma_days
    Current EMA =  
    (Current value * (1 +( Constant /(1+ days of interest)) + (Previous EMA * (1 -( Constant /(1+ days of interest)), constant is a number generally 2
    '''
    datadf['ema'] = datadf['Close'].ewm(span=ma_days,adjust=False).mean()
    
    return datadf


# Choppiness Index Calculation
def choppiness_index(datadf,lookback=14):
    '''
    Calculate Choppiness Index for the trend 
    CI = 100 * log10(sum(Average Range of price over past n steps)/(Highest price during n steps -Lowest price during n steps))) / log10(n), 
    where n_steps= user defined length 
    '''
    # True Range
    length = len(datadf)
    datadf['tr']=0
    for i in range(1,length):
        datadf['tr'][i] = max([datadf['High'][i] -datadf['Low'][i],datadf['High'][i] - datadf['Close'][i-1],datadf['Close'][i-1] - datadf['Low'][i]])
        
    # ATR : Average True Range 
    datadf['ATR']=0
    datadf['ATR'] = datadf['tr'].rolling(lookback).mean()
    
    # ATR Sum
    
    datadf['ATR_sum'] = datadf['ATR'].rolling(lookback).sum()
    
    # Division 
    datadf['temp'] = datadf['ATR_sum'] /(datadf['High'].rolling(lookback).max() - datadf['Low'].rolling(lookback).min())
    
    # Choppiness Index
    datadf['CI']= 100 * np.log10(datadf['temp']) / np.log10(lookback)
    
    datadf = datadf.drop(['tr','ATR','ATR_sum','temp'],axis=1)
    
    return datadf
        
        
def bollinger_bands(datadf,k=2,n_days=14):
    datadf['Bollinger_centralBand'] = datadf['Close'].rolling(n_days).mean()
    
    datadf['Bollinger_upperBand'] = datadf['Bollinger_centralBand'] + k*datadf['Close'].rolling(n_days).std()
    
    datadf['Bollinger_lowerband'] = datadf['Bollinger_centralBand'] - k*datadf['Close'].rolling(n_days).std()
    
    return datadf

