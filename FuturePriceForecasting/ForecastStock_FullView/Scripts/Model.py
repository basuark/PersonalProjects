import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

def buildmodel(datadfs,opt_lr=1e-4,slidingwindow=15):
    data = datadfs.filter(['FibonacciLevels'])
    dataset =data.values
    training_data_len = int(np.ceil(len(dataset)*0.95))
    
    scaler_df=MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler_df.fit_transform(dataset)
    
     # Creating XY_splits
    train_data = scaled_data[0:int(training_data_len),:]
    X_train=[]
    y_train=[]

    for i in range(60,len(train_data)):
      X_train.append(train_data[i-60:i,0])
      y_train.append(train_data[i,0])

    #  if i<=61:
    #    print(X_train,y_train)

    X_train,y_train = np.array(X_train),np.array(y_train)

    X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
    
    # Building  model

    #Setting random seed
    tf.random.set_seed(42)

    # model
    model_lstm_tcs = tf.keras.Sequential([
          tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],1)),
          tf.keras.layers.LSTM(60,return_sequences=True),
          tf.keras.layers.LSTM(60),
          tf.keras.layers.Dense(1)
    ])

    # Compile model
    model_lstm_tcs.compile(loss=tf.keras.losses.mae,
                       optimizer=tf.keras.optimizers.Adam(learning_rate=opt_lr),
                       metrics='mae')

    # Train the model
    model_lstm_tcs.fit(X_train,y_train,epochs=40,verbose=0)
    
         # Creating XY_splits
    test_data = scaled_data[int(training_data_len)-60:,:]
    X_test=[]
    y_test=dataset[training_data_len:,:]

    for i in range(60,len(test_data)):
      X_test.append(test_data[i-60:i,0])

    X_test=np.array(X_test)
    X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    
    y_pred_test = model_lstm_tcs.predict(X_test)
    y_pred_test_real = scaler_df.inverse_transform(y_pred_test)
    
    # Current prediction

    datadf = datadfs.filter(['Date','FibonacciLevels'])

    # Plot the data
    train = datadf[:training_data_len]
    valid = datadf[training_data_len:]
    train = train.set_index('Date')
    valid = valid.set_index('Date')

    valid['Predictions'] = y_pred_test_real
    finaldf = pd.concat([train,valid],axis=0)
    # Visualize the data
    plt.figure(figsize=(10,8))
    finaldf[['FibonacciLevels','Predictions']].plot(figsize=(10,8))
    plt.title('Predictions based on Fibonacci Prices using LSTM model with 2 hidden layers')
    plt.xlabel('Date')
    plt.ylabel('Price in Rs')
    plt.legend(['Train','Val','Predictions'],loc='upper left')

    valid[['FibonacciLevels','Predictions']].plot(figsize=(10,8))
    plt.xlabel('Date')
    plt.title('A Closer Look')
    plt.ylabel('Price in Rs')
    plt.legend(['Test Data','Forecast'],loc='upper left')
    plt.show()
    
    # Repeating last step for predicting future values
    last_60days_data = dataset[dataset.shape[0]-62:-2,:]
    last_60days_data.shape
    
    
    # Scale it
    last_60days_data_scaled= scaler_df.fit_transform(last_60days_data)
    
    ## Create test data
    test_data = last_60days_data_scaled
    X_test_data=[]
    X_test_data.append(test_data[-60:,:])
    X_test_data=np.array(X_test_data)
    X_test_data = np.reshape(X_test_data,(X_test_data.shape[0],X_test_data.shape[1],1))

    # Forecast for 1 day : to repeat use loop by stitching this 1 day
    y_pred_fut = model_lstm_tcs.predict(X_test_data)
    y_pred_fut_real = scaler_df.inverse_transform(y_pred_fut)
    
    

    for i in range(slidingwindow-1):
      test_data=np.append(test_data,y_pred_fut,axis=0)
      X_test_data=[]
      X_test_data.append(test_data[-60:,:])
      X_test_data=np.array(X_test_data)
      X_test_data = np.reshape(X_test_data,(X_test_data.shape[0],X_test_data.shape[1],1))
      # Forecast for 1 day : to repeat use loop by stitching this 1 day
      y_pred_fut = model_lstm_tcs.predict(X_test_data)
      y_pred_fut_real = np.append(y_pred_fut_real,scaler_df.inverse_transform(y_pred_fut),axis=0)
    
    datadfs_copy = datadfs.copy()
    datadfs_copy['Date'] = pd.to_datetime(datadfs_copy['Date'],format='%Y-%m-%d')
    datadfs_copy = datadfs_copy.set_index('Date')
    
    future = pd.DataFrame(data=y_pred_fut_real, columns=['Predictions'],index=pd.bdate_range(start=datadfs_copy.index[-1] + pd.Timedelta(days=1),periods=15))
    future = future.reset_index()
    future.columns=['Date','Predictions']
    #future['Date'] = pd.to_datetime(future['Date'],'%Y-%m-%d')
    future = future.set_index('Date')
    
    finaldf = pd.concat([finaldf,future],axis=0)
    # Visualize the data
    plt.figure(figsize=(10,8))
    finaldf[['FibonacciLevels','Predictions']].plot(figsize=(10,8),grid=True)
    plt.title('Predictions based on Fibonacci Prices using LSTM model with 2 hidden layers')
    plt.xlabel('Date')
    plt.ylabel('Price in Rs')
    plt.legend(['Train','Val','Predictions'],loc='upper left')
    valid = pd.concat([future,valid],axis=0)
    valid[['FibonacciLevels','Predictions']].plot(figsize=(10,8),grid=True)
    plt.xlabel('Date')
    plt.title('A Closer Look')
    plt.ylabel('Price in Rs')
    plt.legend(['Test Data','Forecast'],loc='upper left')
    
    datadfs_copy= datadfs_copy[['High','Low','Trend','level_0','level_38','level_50','level_61','level_78','level_100','level_127','level_161','FibonacciLevels']]
    datadfs_copy = datadfs_copy.dropna()
    
    finaldf=finaldf[1:]

    req_fields=['High','Low','Trend','level_0','level_38','level_50','level_61','level_78','level_100','level_127','level_161','FibonacciLevels','Predictions']
    datadfs_copy['Predictions']=np.nan
    finaldf[['High','Low','Trend','level_0','level_38','level_50','level_61','level_78','level_100','level_127','level_161']]=np.nan
    
    
    final_dataframe = pd.concat([datadfs_copy,finaldf[-20:]],axis=0)

    final_dataframe=final_dataframe.reset_index()
    final_dataframe = final_dataframe.set_index('Date')
    plt.figure(figsize=(20,10))
    plt.plot(final_dataframe[['High','Low']][-100:])
    plt.plot(final_dataframe['Predictions'][-100:])
    plt.plot(final_dataframe['level_0'][-100:])
    plt.plot(final_dataframe['level_38'][-100:])
    plt.plot(final_dataframe['level_50'][-100:])
    plt.plot(final_dataframe['level_61'][-100:])
    plt.plot(final_dataframe['level_78'][-100:])
    plt.plot(final_dataframe['level_100'][-100:])
    plt.plot(final_dataframe['level_127'][-100:])
    plt.plot(final_dataframe['level_161'][-100:])
    pred = final_dataframe['Predictions'][-16:]
    return final_dataframe,pred
    
    