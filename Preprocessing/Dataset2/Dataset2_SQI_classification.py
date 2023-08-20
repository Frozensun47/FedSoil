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
from sklearn.metrics import confusion_matrix,classification_report
from pylab import rcParams
import datetime

df=pd.read_csv("data\Dataset_2_preprocessing\Dataset2_SQI_calculated.csv")

df=df.set_index('NEWSUID')

zeros=[]
for i in range(0,len(df)):
    zeros.append(0)
df['Category']=zeros

df['SQI']= (df['SQI'] - df['SQI'].min())/(df['SQI'].max() - df['SQI'].min())

df['SQI']*=100

for i in range(0,len(df)-1):
    if(df['SQI'][i]<=22):
        df['Category'].iloc[i]=0       
    elif(df['SQI'][i]>22 and df['SQI'][i]<=39):
        df['Category'].iloc[i]=1
    elif(df['SQI'][i]>39 and df['SQI'][i]<=53):
        df['Category'].iloc[i]=2
    elif(df['SQI'][i]>53):
        df['Category'].iloc[i]=3

df['Category'].value_counts()
df=df.drop('SQI',axis=1)

df.to_csv("data\Dataset_2_preprocessing\Dataset2_SQI_classified.csv",index=False)



