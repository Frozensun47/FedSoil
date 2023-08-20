import warnings
warnings.filterwarnings('ignore')
import pandas as pd
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
import datetime
from datetime import datetime
from numpy import log
from sklearn import metrics # for the evaluation
from sklearn.preprocessing import LabelEncoder
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
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("data\Dataset_2_preprocessing\Dataset_2.csv")

(df['PRID'].value_counts()/df['PRID'].count())*100

df=df.drop(['CFRAG','DrainNum','ALSA','CLAF','Drain','Layer','TopDep','BotDep','PSCL','ELCO_std','GYPS_std','PHAQ_std','ESP_std','ALSA_std','BSAT_std','TEB_std','ECEC_std','CECc_std','CECS_std','CNrt_std','TOTN_std','ORGC_std','TAWC_std','BULK_std','CLPC_std','STPC_std','SDTO_std','CFRAG_std','PROP','TCEQ_std'],axis=1)

df=df.dropna(how='all')
df=df.fillna(method='bfill', axis=0)
df=df.fillna(method='ffill', axis=0)
df.dropna(inplace=True)

cols=[]
for i in df.columns:
    cols.append(i)
cols

df = df[df['BULK'] >= 0]
df=df[df['ORGC']>=0]
df=df[df['ECEC']>=0]
df=df[df['ELCO']>=0]
df=df[df['TAWC']>=0]
#df=df[df['ALSA']>=0]
#df=df[df['TCEQ']>=0]

df['BULK']=df.BULK.interpolate(limit_direction='both')
df['ORGC']=df.ORGC.interpolate(limit_direction='both')
df['ECEC']=df.ECEC.interpolate(limit_direction='both')
df['ELCO']=df.ELCO.interpolate(limit_direction='both')
#df['ALSA']=df['ALSA'].replace(0,df['ALSA'].mean())
df['TCEQ']=df.TCEQ.interpolate(limit_direction='both')
df['ESP']=df.ESP.interpolate(limit_direction='both')
#df['GYPS']=df['GYPS'].replace(0,df['GYPS'].mean())

df['VWC']=df['TAWC']/100 +0.21

df=df[df['VWC']<=0.45]

df.to_csv("data\Dataset_2_preprocessing\Dataset2_unprocessed_1.csv",index=False)

df=pd.read_csv("data\Dataset_2_preprocessing\Dataset2_unprocessed_1_2.csv")

df.set_index('NEWSUID', inplace=True)

df.drop_duplicates(inplace=True,keep='first')
df=df.drop(['SCID','PRID'],axis=1)

X=[]
for i in range(len(df)):
    X.append(i)

x=[]
x=df.columns
x

fig, axes = plt.subplots(len(df.columns),1, figsize=(20,30))
ct=0
for i in x:
    axes[ct].plot(X,df[i])
    axes[ct].set_title (str(i),loc='left')
    ct+=1

fig, axes = plt.subplots(len(df.columns),1, figsize=(20,30))
ct=0
for i in x:
    axes[ct].plot(X,df[i])
    axes[ct].set_title (str(i),loc='left')
    ct+=1

df.to_csv("data\Dataset_2_preprocessing\Dataset2_processed.csv")

