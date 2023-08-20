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
import datetime
from datetime import datetime
from numpy import log
from sklearn import metrics # for the evaluation
from sklearn.preprocessing import LabelEncoder
from numpy.linalg import eig
import tensorflow as tf

def preprocess(df):    
    scaler = StandardScaler()
    scaler.fit(df)
    df_norm = scaler.transform(df)    
    return df_norm

df=pd.read_csv("data\Dataset_2_preprocessing\dataset1_processed.csv")

df.set_index('NEWSUID', inplace=True)

df2=df[['ELCO','VWC','ORGC','BULK','PHAQ','ECEC','TOTN','ESP','TCEQ','DP','Temp']].copy()

scaled_df=preprocess(df.T)

pca=PCA()

pca.fit(scaled_df)

pca_data=pca.transform(scaled_df)
percentage_varriation = np.round(pca.explained_variance_ratio_*100,decimals=1)

labels = ['PC'+str(x) for x in range(1,len(percentage_varriation)+1)]

plt.bar(x=range(1,len(percentage_varriation)+1),height=percentage_varriation,tick_label=labels)
plt.ylabel('Percentage of Explained Varriance')
plt.xlabel('Principal Component')
plt.title('Screen Plot')
plt.show()

pca_df=pd.DataFrame(pca_data,columns=labels,index=df.columns)

fig=plt.figure(figsize=(10,10))
plt.scatter(pca_df.PC1,pca_df.PC2)
plt.title('PCA GRAPH')
plt.xlabel('PC1 - {0}%'.format(percentage_varriation[0]))
plt.ylabel('PC2 - {0}%'.format(percentage_varriation[1]))
#plt.zlabel('PC3 - {0}%'.format(percentage_varriation[2]))
for sample in pca_df.index:
    plt.annotate(sample,(pca_df.PC1.loc[sample],pca_df.PC2.loc[sample]))
plt.grid(True)

print(pca.n_features_)
print(pca_df)

fig, ax = plt.subplots(figsize=(30,30)) 
ax = sns.heatmap(pca_df, annot=True, cmap='Spectral')
plt.show()
covMatrix = pd.DataFrame(np.cov(scaled_df),columns=df.columns,index=df.columns)
fig1, ax = plt.subplots(figsize=(30,30)) 
ax = sns.heatmap(covMatrix, annot=True, cmap='Spectral')
plt.show()

scaled_df_pd= pd.DataFrame(scaled_df,index=df.columns)

fig3, ax = plt.subplots(figsize=(30,30)) 
ax = sns.heatmap((scaled_df_pd.T).cov(), annot=True, cmap='Spectral')
plt.show()

eigenvalues,eigenvectors=eig(np.cov(scaled_df))
eigenvalues=pd.DataFrame(eigenvalues,index=df.columns,columns=['Eigenvalue'])
eigenvalues.sort_values('Eigenvalue',ascending=False,inplace=True)
eigenvalues.round(decimals=5)

correlationMatrix = pd.DataFrame(np.corrcoef(scaled_df),columns=df.columns,index=df.columns)
fig2, ay = plt.subplots(figsize=(30,30)) 
ay = sns.heatmap(correlationMatrix, annot=True, cmap='Spectral')
plt.show()

df=df.drop(['TAWC','CNrt','CECc','ECEC','TEB','ESP'],axis=1)

df.to_csv("data\Dataset_2_preprocessing\Dataset2_processed_final.csv")

