from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean
from sklearn.metrics import silhouette_score
import seaborn as sns
# from sklearn.model_selection import KMeans
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
d=load_iris()
print(d.feature_names)
df=pd.DataFrame(d.data)
# df.drop[['sepal width'],['sepal length']]
df.columns=d.feature_names
df.drop('sepal length (cm)',axis='columns',inplace=True)
df.drop('sepal width (cm)',axis='columns',inplace=True)
# df.drop(['sepal width (cm)','sepal length (cm)'],axis=1)
# print(df)
scaler=MinMaxScaler()
# scaler.fit(df[['petal width (cm)']])
# # scaler.fit(df[['petal length (cm)']])
# # print(df['petal width (cm)'])
# # print(df['petal length (cm)'])
# df['petal width (cm)']=scaler.transform(df[['petal width (cm)']])
# df['petal length (cm)']=scaler.transform(df[['petal length (cm)']])
# plt.scatter(df['petal length (cm)'],df['petal width (cm)'])
# plt.show()
# k_range=1,11
# wcss=[]
# for i in range(1,11):
#     km=KMeans(n_clusters=i)
#     km.fit(df[['petal length (cm)']],df[['petal width (cm)']])
#     wcss.append(km.inertia_)
# # print(wcss)
# plt.scatter(range(1,11),wcss)
# plt.show()
km=KMeans(n_clusters=3)
# km.fit(df[['petal length (cm)','petal width (cm)']])
y_pre=km.fit_predict(df[['petal length (cm)','petal width (cm)']])
df['cluster']=y_pre
# print(df)
print('kmeans: {}'.format(silhouette_score(df, km.labels_,metric='euclidean'))) 
# print('kmeans: {}'.format(silhouette_score(df, kmeans.labels_, 
                                          
clusters=km.cluster_centers_
print(clusters)
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

# df4=df[df.cluster==3]
# # df5=df(df.cluster==4)
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='red')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='green')
plt.scatter(df3['petal length (cm)'],df3['petal width (cm)'],color='yellow')
# # plt.scatter(df4['petal width (cm)'],df4['petal length (cm)'],color='blue')
# # plt.scatter(df5['petal width (cm)'],df5['petal length (cm)'],color='red')
plt.scatter(clusters[0][0],clusters[0][1],marker='*',color='purple')
plt.scatter(clusters[1][0],clusters[1][1],marker='*',color='purple')
plt.scatter(clusters[2][0],clusters[2][1],marker='*',color='purple')

scaler.fit(df[['petal width (cm)']])
scaler.fit(df[['petal length (cm)']])
print(df['petal width (cm)'])
print(df['petal length (cm)'])
df['petal width (cm)']=scaler.transform(df[['petal width (cm)']])
df['petal length (cm)']=scaler.transform(df[['petal length (cm)']])
k_range=1,11
wcss=[]
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(df[['petal length (cm)']],df[['petal width (cm)']])
    wcss.append(km.inertia_)
print(wcss)
plt.scatter(range(1,11),wcss)
# plt.show()


