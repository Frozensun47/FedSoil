import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from numpy import log
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from kneed import KneeLocator

import sklearn.metrics as sm # for evaluating the model
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import scale 
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report



import datetime
from datetime import datetime
#from statsmodels.tsa.stattools import adfuller


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Flatten
from keras.callbacks import EarlyStopping
from keras.models import load_model


df=pd.read_csv('data\Dataset_1_preprocessing\Dataset_1.csv')

df=df.drop(['Datetime Slot'],axis=1)

df=df.rename(columns={'Actual Datetime': 'date_time','Device ID':'deviceId','Base Station':'Base','Soil Temperature (°C)':'temp','Electrical Conductivity (µS/m)':'EC','Dielectric Permittivity':'DP','Volumetric Water Content':'VWC'})

df=df.rename(columns={'Actual Datetime': 'date_time','Device ID':'deviceId','Base Station':'Base','Soil Temperature (°C)':'temp','Electrical Conductivity (µS/m)':'EC','Dielectric Permittivity':'DP','Volumetric Water Content':'VWC'}) #rename the rows for simplicity
df=df.dropna(how='all')#remove all the rows which have all NaN values
df=df.fillna(method='bfill', axis=0)
df=df.fillna(method='ffill', axis=0)
df=df.dropna()
df['date_time']=df['date_time'].replace({'-06:00':''},regex=True) #clean the date - time
df['date_time']=df['date_time'].replace({'-05:00':''},regex=True) #clean the date - time

fig = px.scatter(df, x="date_time", y="Base", title="date_time vs Base") 
#fig.show()

(df['Base'].value_counts()/df['Base'].count())*100

df=df.loc[df['Base']=='66ED'] #only the data with base=66ED taken

df=df.drop('Base',axis=1)

fig = px.scatter(df, x="date_time", y="deviceId", title="date_time vs DeviceId") 
#fig.show()

(df['deviceId'].value_counts()/df['deviceId'].count())*100

df[df['deviceId']=='3DE868'].count() #==37890

(df['deviceId'].value_counts()/39042)*100

df2 = df['deviceId'].value_counts().rename_axis('deviceId').reset_index(name='Percent_device')
df2['Percent_device']/=39042
df2['Percent_device']*=100
df=pd.merge(df,df2,on='deviceId',how='left')
df

df=df.loc[df['Percent_device']>50] #device with more than 50% data 
df.reset_index(inplace=True)
df

fig = px.line(df, x="date_time", y="temp",color='deviceId', title="B1 date_time vs temp") 
#fig.show()

count_zero_Temp = (df['temp']==0).sum()
print(count_zero_Temp)
fig = px.line(df, x="date_time", y="EC",color='deviceId', title="B1 date_time vs Electrical Conductivity") 
#fig.show()

fig = px.line(df, x="date_time", y="VWC",color='deviceId', title="B1 date_time vs Volumetric Water Content") 
#fig.show()

df=df.loc[df['VWC']<0.45]
df.describe()

count_zero_VWC = (df['VWC']==0).sum()
print(count_zero_VWC)

fig = px.line(df, x="date_time", y="DP",color='deviceId', title="B1 Date_time vs Dielectric Permittivity") 
#fig.show()

count_zero_DP = (df['DP']==0).sum()
print(count_zero_DP)

df['date_parsed'] = pd.to_datetime(df['date_time'], format="%Y-%m-%d %H:%M:%S")

df['hour']=df['date_parsed'].dt.hour
df['day']=df['date_parsed'].dt.day
df['year']=df['date_parsed'].dt.year
df['month']=df['date_parsed'].dt.month
df.head()

df['deviceId'].value_counts()

df.to_csv("data\Dataset_1_preprocessing\Processed_data_temp.csv",index=False)

df=pd.read_csv("data\Dataset_1_preprocessing\Processed_data_temp_2.csv")

df=df.drop('index',axis=1)

client1=df[df['deviceId']=='3DE868'].copy()
client1=client1.groupby(['year','month','day'],sort=False,group_keys=False).mean()
client1.to_csv("data\Dataset_1_preprocessing\Processed_data_client1.csv",index=False)
client1

client2=df[df['deviceId']=='3DF593'].copy()
client2=client2.groupby(['year','month','day'],sort=False,group_keys=False).mean()
client2.to_csv("data\Dataset_1_preprocessing\Processed_data_client2.csv",index=False)
client2

client3=df[df['deviceId']=='3DFF9A'].copy()
client3=client3.groupby(['year','month','day'],sort=False,group_keys=False).mean()
client3.to_csv("data\Dataset_1_preprocessing\Processed_data_client1.csv",index=False)
client3

client4=df[df['deviceId']=='3DF675'].copy()
client4=client4.groupby(['year','month','day'],sort=False,group_keys=False).mean()
client4.to_csv("data\Dataset_1_preprocessing\Processed_data_client1.csv",index=False)
client4

client5=df[df['deviceId']=='3DFFF0'].copy()
client5=client5.groupby(['year','month','day'],sort=False,group_keys=False).mean()
client5.to_csv("data\Dataset_1_preprocessing\Processed_data_client1.csv",index=False)
client5

client6=df[df['deviceId']=='3DFF73'].copy()
client6=client6.groupby(['year','month','day'],sort=False,group_keys=False).mean()
client6.to_csv("data\Dataset_1_preprocessing\Processed_data_client1.csv",index=False)
client6

client7=df[df['deviceId']=='3DE9BB'].copy()
client7=client7.groupby(['year','month','day'],sort=False,group_keys=False).mean()
client7.to_csv("data\Dataset_1_preprocessing\Processed_data_client1.csv",index=False)
client7

df=df.groupby(['year','month','day'],sort=False,group_keys=False).mean()

df=df.reset_index()

df.to_csv("data\Dataset_1_preprocessing\Processed_data.csv",index=False)