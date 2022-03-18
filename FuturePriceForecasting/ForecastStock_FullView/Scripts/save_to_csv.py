import pandas as pd

def save_file(df,name):
    df.to_csv('data/'+name+'.csv')
    print("File "+name+'.csv'+" has been saved")
