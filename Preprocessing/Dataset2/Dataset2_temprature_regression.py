import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from kneed import KneeLocator
import sklearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import plotly.graph_objects as go
from sklearn.preprocessing import scale # for scaling the data
import sklearn.metrics as sm # for evaluating the model
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import datetime
from datetime import datetime
from numpy import log
from sklearn import metrics # for the evaluation
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from numpy.linalg import eig
import tensorflow as tf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential,model_from_json
from tensorflow.keras.layers import Dense,Softmax,Dropout
from tensorflow.keras.layers import LSTM,Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing
import math
import uuid
import random
import zipfile


def preprocess1(df):    
    scaler = MinMaxScaler()
    scaler.fit(df)
    df_norm = scaler.transform(df)    
    return df_norm
def preprocess2(df):    
    scaler = StandardScaler()
    scaler.fit(df)
    df_norm = scaler.transform(df)    
    return df_norm

df=pd.read_csv("data\Dataset_1_preprocessing\Processed_data.csv")

df=df[['temp', 'EC', 'DP', 'VWC']].copy()

scaled_df=df

scaled_df=pd.DataFrame(scaled_df,columns=df.columns)

cols=scaled_df.columns
cols

#X = scaled_df.drop(['SQI'],axis=1)
#X = scaled_df.iloc[:, :12].copy()
X=df[['VWC']].copy()
y = df[['temp']].copy()
print(y.dtypes)
print(X.dtypes)

X_train, X_val, y_train, y_val = train_test_split(X, y,test_size = 0.2,shuffle=False)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor.fit(X_train,y_train)
y_prediction_RFR =  regressor.predict(X_val)
print(regressor.score(X_train,y_train))
score=r2_score(y_val,y_prediction_RFR )
print('r2 socre is ',score)
print('mean_sqrd_error is ',mean_squared_error(y_val,y_prediction_RFR ))
print('root_mean_squared error of is ',np.sqrt(mean_squared_error(y_val,y_prediction_RFR )))
plt.figure(figsize=(50,10))
plt.plot(list(range(0, len(y_val))),y_val ,color='blue')         
plt.plot(list(range(0, len(y_val))), y_prediction_RFR, color='black')
plt.show()

LR = LinearRegression()
LR.fit(X_train,y_train)
print(LR.score(X_train,y_train))
y_prediction =  LR.predict(X_val)
score=r2_score(y_val,y_prediction)
print('r2 socre is ',score)
print('mean_sqrd_error is ',mean_squared_error(y_val,y_prediction))
print('root_mean_squared error of is ',np.sqrt(mean_squared_error(y_val,y_prediction)))
plt.figure(figsize=(50,10))
plt.plot(list(range(0, len(y_val))),y_val ,color='blue')         
plt.plot(list(range(0, len(y_val))), y_prediction, color='black')
plt.show()

svrr = SVR(kernel='rbf')
svrr.fit(X_train,y_train)
print(svrr.score(X_train,y_train))
y_prediction =  svrr.predict(X_val)
score=r2_score(y_val,y_prediction)
print('r2 socre is ',score)
print('mean_sqrd_error is ',mean_squared_error(y_val,y_prediction))
print('root_mean_squared error of is ',np.sqrt(mean_squared_error(y_val,y_prediction)))
plt.figure(figsize=(50,10))
plt.plot(list(range(0, len(y_val))),y_val ,color='blue')         
plt.plot(list(range(0, len(y_val))), y_prediction, color='black')
plt.show()

df2=pd.read_csv("data\Dataset_2_preprocessing\Dataset2_unprocessed_1.csv")

to_predict = df2[['VWC']]

y2_prediction =  svrr.predict(to_predict)
y2_prediction

df2['Temp']=y2_prediction

df2.to_csv("data\Dataset_2_preprocessing\Dataset2_unprocessed_1_1.csv",index=False)






























