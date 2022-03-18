import pandas as pd
import numpy as np

def get_swings(datadf,windowsize=12):
    # Find 14 days High
    length = len(datadf)
    datadf['SwingHigh'] = np.nan
    datadf['SwingLow'] = np.nan
    datadf['Trend']=0
    highindiceslist=[]
    highindiceslistsmooth=[]
    for i in range(0,length,windowsize):
        maxvalue= datadf['High'][i:i+windowsize].max()
        idx=datadf.index
        highindex= idx[datadf['High']==maxvalue]
        highindiceslist.append(highindex[0])
    
    for i in range(len(highindiceslist)-1):
        if highindiceslist[i]+1 ==highindiceslist[i+1]:
            pass
        else:
            highindiceslistsmooth.append(highindiceslist[i])
    highindiceslist=highindiceslistsmooth 
    highindiceslist.append(datadf.index[-1])
    for i in range(0,len(highindiceslist)-1):
        datadf['SwingHigh'][highindiceslist[i]:highindiceslist[i+1]] = datadf['High'][highindiceslist[i]]
        datadf['SwingLow'][highindiceslist[i]:highindiceslist[i+1]+1]= datadf['Low'][highindiceslist[i]:highindiceslist[i+1]+1].min()
        if datadf['High'][highindiceslist[i]] > datadf['High'][highindiceslist[i+1]] :
            # downtrend for current trend
            datadf['Trend'][highindiceslist[i]:highindiceslist[i+1]]=-1
        elif datadf['High'][highindiceslist[i]] < datadf['High'][highindiceslist[i+1]]:
            datadf['Trend'][highindiceslist[i]:highindiceslist[i+1]]=1
    return datadf,highindiceslist

def adjust_swings(datadf,indices):
    datadf['AdjSwingHigh']=np.nan
    datadf['AdjSwingLow']=np.nan
    for i in range(len(indices)-2):
        datadf['AdjSwingHigh'][indices[i+1]:indices[i+2]] = max(datadf['SwingHigh'][indices[i]],datadf['SwingHigh'][indices[i+1]])
        datadf['AdjSwingLow'][indices[i+1]:indices[i+2]] = min(datadf['SwingLow'][indices[i]],datadf['SwingLow'][indices[i+1]])
    return datadf
    

def get_fibonacciLevels(datadf):
    #levels=['level_0L','level_38L','level_50L','level_61L','level_76L','level_78L','level_100L','level_127L',
    #        'level_0H','level_38H','level_50H','level_61H','level_76H','level_78H','level_100H','level_127H']
    levels=['level_0','level_38','level_50','level_61','level_78','level_100','level_127','level_161','level_261']
    for i in levels:
        datadf[i]=np.nan
    datadf['val']=np.nan
    datadf['diff'] = abs(datadf['AdjSwingHigh']-datadf['AdjSwingLow'])
    length=len(datadf)
    for i in range(length):
        swinghigh=datadf['AdjSwingHigh'][i]
        high=datadf['High'][i]
        low=datadf['Low'][i]
        swinglow=datadf['AdjSwingLow'][i]
        diff=datadf['diff'][i]
        if datadf['Trend'][i]==-1:
            # down
            datadf['level_0'][i] = swinghigh
            datadf['level_38'][i] = swinghigh - (diff*38.2/100)
            datadf['level_50'][i] = swinghigh - (diff*50/100)
            datadf['level_61'][i] = swinghigh - (diff*61.8/100)
            #datadf['level_76'][i] = swinghigh - (diff*76.4/100)
            datadf['level_78'][i] = swinghigh - (diff*78.6/100)
            datadf['level_100'][i] = swinglow
            datadf['level_127'][i] = swinghigh - (diff*127/100)
            datadf['level_161'][i] = swinghigh - (diff*161.8/100)
            datadf['level_261'][i] = swinghigh - (diff*261.8/100)
            datadf['val'][i]=low
            
        elif datadf['Trend'][i]==1:
            datadf['level_0'][i] = swinglow
            datadf['level_38'][i] = swinglow + (diff*38.2/100)
            datadf['level_50'][i] = swinglow + (diff*50/100)
            datadf['level_61'][i] = swinglow + (diff*61.8/100)
            #datadf['level_76'][i] = swinglow + (diff*76.4/100)
            datadf['level_78'][i] = swinglow + (diff*78.6/100)
            datadf['level_100'][i] = swinghigh
            datadf['level_127'][i] = swinglow + (diff*127/100)
            datadf['level_161'][i] = swinglow + (diff*161.8/100)
            datadf['level_261'][i] = swinglow + (diff*261.8/100)
            datadf['val'][i]=high
            
        elif datadf['Trend'][i]==0:
            datadf['val'][i]=datadf['Close'][i]

        if i==length-1 :
          datadf['level_0'][i] = 		datadf['level_0'][i-1] 
          datadf['level_38'][i] =     datadf['level_38'][i-1] 
          datadf['level_50'][i] =     datadf['level_50'][i-1] 
          datadf['level_61'][i] =     datadf['level_61'][i-1] 
          #datadf['level_76'][i] =    #datadf['level_76'][i-1]
          datadf['level_78'][i] =     datadf['level_78'][i-1] 
          datadf['level_100'][i] =    datadf['level_100'][i-1]
          datadf['level_127'][i] =    datadf['level_127'][i-1]
          datadf['level_161'][i] =    datadf['level_161'][i-1]
          datadf['level_261'][i] =    datadf['level_261'][i-1]

        
    return datadf
    

def calculate_fiblevels(datadf,indices):
    datadf=datadf.fillna(0)
    length = len(datadf)
    datadf['FibonacciLevels']=np.nan
    levels=['level_0','level_38','level_50','level_61','level_78','level_100','level_127','level_161','level_261']
    for i in range(length):
        
        if  datadf['High'][i] > datadf['level_161'][i] :
            datadf['FibonacciLevels'][i]= datadf['High'][i]
        elif datadf['Low'][i] <datadf['level_161'][i] :
            datadf['FibonacciLevels'][i] = datadf['Low'][i]
        else:
            nearestlevelidx=""
            min_diff=999
            for j,k in enumerate(levels):
                diff =abs(datadf[k][i]-datadf['val'][i])
                if diff <min_diff:
                    min_diff=diff
                    nearestlevelidx=k
            #print(nearestlevelidx,min_diff)
            datadf['FibonacciLevels'][i] = datadf[nearestlevelidx][i]
            
    return datadf