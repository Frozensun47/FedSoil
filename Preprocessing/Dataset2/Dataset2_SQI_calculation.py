import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
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
from sklearn.model_selection import train_test_split
from numpy.linalg import eig
import tensorflow as tf

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

df=pd.read_csv("data\Dataset_2_preprocessing\Dataset2_processed_final.csv")

df.set_index('NEWSUID', inplace=True)

scaled_df=preprocess1(df)

scaled_df=pd.DataFrame(scaled_df,columns=df.columns)

cols=scaled_df.columns
cols

weight=[0.0206,0.0344,0.2274,0.0553,0.1267,0.0793,0.0875,0.0625,0.0376,0.0187,0.0661,0.1641,0.0099,0.0099]
#eight = [x+0.00165 for x in weight]
np.sum(weight)

j=0
for i in cols:
    scaled_df[i]=weight[j]*scaled_df[i]

scaled_df['SQI']=scaled_df[scaled_df.columns].sum(axis=1)

df=df.reset_index()

df['SQI']=scaled_df['SQI']

df.to_csv("data\Dataset_2_preprocessing\Dataset2_SQI_calculated.csv",index=False)

