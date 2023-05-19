import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize,MinMaxScaler,StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sh
# %matplotlib inline
d=pd.read_csv('Wholesale customers data.csv')
# print(d)
data_scaled = normalize(d)
data_scaled = pd.DataFrame(data_scaled, columns=d.columns)
# print(data_scaled)
# scaler=MinMaxScaler()
# scaler.fit(d)
# d1=scaler.transform(d)
# print(d1)

plt.figure(figsize=(11, 8))  
plt.title("Dendrogram")  
dend = sh.dendrogram(sh.linkage(data_scaled, method='ward'))

plt.axhline(y=6, color='g', linestyle='--')
# plt.show()
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
group=cluster.fit_predict(data_scaled)
data_scaled['category']=group
# print(data_scaled)
print(cluster.labels_)
plt.figure(figsize=(10, 7))  
plt.scatter(data_scaled['Fresh'], data_scaled['Grocery'], c=cluster.labels_) 
# plt.show()
# scaler=StandardScaler()
# scaler.fit(d)
# d1=scaler.transform(d)
# d1 = pd.DataFrame(d1, columns=d.columns)
# plt.figure(figsize=(11, 8))  
# plt.title("Dendrogram")  
# dend = sh.dendrogram(sh.linkage(d1, method='ward'))

# # plt.axhline(y=6, color='g', linestyle='--')
# # plt.show()
# cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
# group=cluster.fit_predict(d1)
# # d1['category']=group
# # print(data_scaled)
# print(cluster.labels_)
# plt.figure(figsize=(10, 7))  
# plt.scatter(d1['Fresh'], d1['Grocery'], c=cluster.labels_) 
plt.show()