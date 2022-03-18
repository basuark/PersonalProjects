import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
from Scripts import *
import tensorflow as tf
from Scripts.technicalindicators import *

def Forecast1(ticker):
    output = forecast(ticker)
    output['NewFibLevel'] = output['FibonacciLevels'].shift(-1)
    output = rsi(output, windowdays=14)

    output = stochastic_RSI(output,windowsize=14)

    output = macd(output)

    output = current_EMA(output)

    output = choppiness_index(output)

    output = bollinger_bands(output)
    # Save it 
    output.to_csv('data/'+ticker+'.csv',index=False)
    output=pd.read_csv('data/'+ticker+'.csv')


    output['Date'] = pd.to_datetime(output['Date'],format='%Y-%m-%d')
    output = output.set_index('Date')

    output = output[26:]

    df_final = output[['Volume','Trend','level_0','level_38','level_50','level_61','level_78','level_100','level_127','level_161','FibonacciLevels','RSI','stockRSI','macd_strength','macd','macd_h','ema','CI','Bollinger_centralBand','Bollinger_upperBand','Bollinger_lowerband','NewFibLevel']].copy()

    model_df = df_final[['Volume','Trend','FibonacciLevels','RSI','stockRSI','macd_strength','macd','macd_h','ema','CI','Bollinger_centralBand','Bollinger_upperBand','Bollinger_lowerband','NewFibLevel']]

    model_df= model_df.drop(['macd_h','ema','macd_strength'],axis=1)
    model_df['Gap_BBands'] = model_df['Bollinger_upperBand'] - model_df['Bollinger_lowerband']
    model_df = model_df.drop(['Bollinger_upperBand','Bollinger_lowerband','Bollinger_centralBand'],axis=1)

    model_df = model_df.drop(['macd','stockRSI'],axis=1)

    training_data_len = 60*(model_df.shape[0]//60)+1

    data = model_df.filter(['FibonacciLevels', 'RSI', 'CI','Volume', 'Trend','Gap_BBands','NewFibLevel']).copy()

    # Normalizing data manually
    minn={}
    maxx={}

    for i in data.columns:
        minn[i]=data[i].min()
        maxx[i]=data[i].max()
        data[i+'_scaled'] = (data[i]-minn[i])/(maxx[i]-minn[i])
    data= data.drop(['FibonacciLevels', 'RSI', 'CI','Volume', 'Trend','Gap_BBands','NewFibLevel'],axis=1)
    
    scaled_data = data.values
     # Creating XY_splits
    train_data = scaled_data[0:int(training_data_len),:]
    X_train=[]
    y_train=[]

    for i in range(60,len(train_data)):
        X_train.append(train_data[i-60:i,0:6])
        y_train.append(train_data[i,-1])

    #  if i<=61:
    #    print(X_train,y_train)

    X_train,y_train = np.array(X_train),np.array(y_train)

    X_train_new = np.reshape(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2],1))

    # Setting random seed
    tf.random.set_seed(42)

    # Model1
    model_lstm = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train_new.shape[1],1)),
        tf.keras.layers.LSTM(570,return_sequences=True),
        tf.keras.layers.LSTM(200),
        tf.keras.layers.Dense(1,activation='relu')
    ])

    # compile
    model_lstm.compile(loss=tf.keras.losses.mae,
                    optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                    metrics=['mae'])

    # Fitting model
    model_lstm_fit = model_lstm.fit(X_train_new,y_train,epochs=20,verbose=0)

    last_60days_data = scaled_data[scaled_data.shape[0]-61:-1,:]
    # Create test data
    test_data = last_60days_data
    X_test_data=[]
    X_test_data.append(test_data[-60:,0:6])
    X_test_data=np.array(X_test_data)

    X_test_data = np.reshape(X_test_data,(X_test_data.shape[0],X_test_data.shape[1]*X_test_data.shape[2],1))

    y_pred_fut = model_lstm.predict(X_test_data)

    y_pred_fut_real = (y_pred_fut *(maxx['NewFibLevel']-minn['NewFibLevel'])) + minn['NewFibLevel']

    loss=min(model_lstm_fit.history['loss'])

    return round(float(y_pred_fut_real),2),loss





