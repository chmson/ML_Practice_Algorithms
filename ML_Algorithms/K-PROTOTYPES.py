
from pickle import FALSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean
import seaborn as sns
from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# from kmodes.kprototypes import KPrototypes

d=pd.read_csv("C:\mlfiles\songs.csv")
df=d[['danceability','energy','loudness','speechiness','acousticness','instrumentalness','valence','tempo']]
wcss=[]
ran=1,20
for i in range(1,20):
    km=KMeans(n_clusters=i,init="k-means++",random_state=40).fit(df)
    wcss.append(km.inertia_)

print(wcss)

# # plt.scatter(range(1,20),wcss)
# scaler=MinMaxScaler()
# scaler.fit(df[['tempo']])
# scaler.fit(df[['loudness']])
# df['tempo']=scaler.transform(df[['tempo']])
# df['loudness']=scaler.transform(df[['loudness']])


km=KMeans(n_clusters=5,init="k-means++",random_state=40)
pre=km.fit_predict(df)
df['clusters']=pre
print('kmeans: {}'.format(silhouette_score(df, km.labels_,metric='euclidean')))

# plt.scatter(pre.head(20),wcss)
# print(df)

clusters=km.cluster_centers_

# print(clusters)
# for i in clusters:
# plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
    # plt.scatter(clusters[:,0],clusters[:,1], s = 80, color = 'green')
# sns.pairplot(df,vars = ['danceability','energy','loudness','speechiness','acousticness','instrumentalness','valence','tempo'])
# plt.legend()
# plt.show()

plt.figure(figsize = (14,5))

sns.countplot(data=df)
df.color.value_counts(sort=False)
plt.show()

# plt.scatter(clusters[2][0],clusters[2][1],marker='*',color='purple')
# d1=d[d.annualspendingcluster==0]
# d2=d[d.annualspendingcluster==1]
# d3=d[d.annualspendingcluster==2]
# d4=d[d.annualspendingcluster==3]
# d5=d[d.annualspendingcluster==4]
# plt.scatter(d1['Annual Income (k$)'],d1['Spending Score (1-100)'],color='red')
# plt.scatter(d2['Annual Income (k$)'],d2['Spending Score (1-100)'],color='green')
# plt.scatter(d3['Annual Income (k$)'],d3['Spending Score (1-100)'],color='yellow')
# plt.scatter(d4['Annual Income (k$)'],d4['Spending Score (1-100)'],color='blue')
# plt.scatter(d5['Annual Income (k$)'],d5['Spending Score (1-100)'],color='pink')
# plt.show()