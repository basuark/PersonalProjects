from .save_to_csv import save_file
from .FibonacciCalculations import *
import matplotlib.pyplot as plt
from .Model import buildmodel
import pandas as pd
from .download_ticker import download_price_history

def forecast(ticker):
    history=download_price_history(ticker)
    save_file(history,ticker)
    history= pd.read_csv('data/'+ticker+'.csv')
    history_new,indices = get_swings(history)
    history_new =adjust_swings(history_new,indices)
    history_fib=get_fibonacciLevels(history_new)
    history_fib =calculate_fiblevels(history_fib,indices)
    save_file(history_fib,'history_fibonacci'+ticker)

    #x = history_fib['Date'][20:500]
    #y1= history_fib['Open'][20:500]
    ##y2= apollohosp_fib['Close'][0:500]
    #y3= history_fib['High'][20:500]
    #y4= history_fib['Low'][20:500]
    #y5 = history_fib['FibonacciLevels'][20:500]
    #
    #plt.figure(figsize=(12,3))
    #plt.plot(x,y1,'g')
    #plt.plot(x,y3,'y')
    #plt.plot(x,y4,'b')
    #plt.plot(x,y5,'r')
    #plt.show()
    #
    #output = buildmodel(history_fib)
    #save_file(output,'Forecast'+ticker)
    return history_fib
    