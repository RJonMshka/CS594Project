import numpy as np
import pandas as pd
from config import *

def load_dataset(*, file_name: str) -> pd.DataFrame:
    data = pd.read_csv(file_name)
    return data

def data_split(df,start,end):
    data = df[(df.datadate >= start) & (df.datadate < end)]
    data = data.sort_values(['datadate','tic'],ignore_index=True)
    data.index = data.datadate.factorize()[0]
    return data

def calculate_price(df):
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)
    data['adjcp'] = data['prccd'] / data['ajexdi']
    data['open'] = data['prcod'] / data['ajexdi']
    data['high'] = data['prchd'] / data['ajexdi']
    data['low'] = data['prcld'] / data['ajexdi']
    data['volume'] = data['cshtrd']
    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def preprocess_data():
    df = load_dataset(file_name=DATA_FILE)
    df = df[df.datadate>=20090000]
    # calcualte adjusted price
    df_preprocess = calculate_price(df)
    df_preprocess.fillna(method='bfill',inplace=True)
    return df_preprocess

def add_turbulence(df):
    turbulence_index = calculate_turbulence(df)
    df = df.merge(turbulence_index, on='datadate')
    df = df.sort_values(['datadate','tic']).reset_index(drop=True)
    return df



def calculate_turbulence(df):
    df_price_pivot=df.pivot(index='datadate', columns='tic', values='adjcp')
    unique_date = df.datadate.unique()
    # start after a year
    start = 252
    turbulence_index = [0]*start
    count=0
    for i in range(start,len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
        cov_temp = hist_price.cov()
        current_temp=(current_price - np.mean(hist_price,axis=0))
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
        if temp>0:
            count+=1
            if count>2:
                turbulence_temp = temp[0][0]
            else:
                turbulence_temp=0
        else:
            turbulence_temp=0
        turbulence_index.append(turbulence_temp)
    turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index, 'turbulence':turbulence_index})
    return turbulence_index



