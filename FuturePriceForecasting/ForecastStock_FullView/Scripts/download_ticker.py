import yfinance as yf
import pandas as pd

def download_price_history(tickerstrng,period='1y',interval='1d'):
    '''
    This function downloads history from yahoo finance
    
    Arguments: 
    tickerstrng : Ticker symbol
    period : The period for which historical price will be downloaded for the ticker, default = 10y
    interval: The interval of price data, default = 1d
    '''
    tickobj = yf.Ticker(tickerstrng)
    tickhist = tickobj.history(period=period, interval=interval)
    return tickhist