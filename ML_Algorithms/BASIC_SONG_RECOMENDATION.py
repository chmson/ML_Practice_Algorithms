# import _DataType
# import http.client
# from http.client import _DataType
from turtle import color
import pandas as pd
import difflib  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import warnings
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
warnings.filterwarnings('ignore')
from matplotlib import style
from statistics import mean
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns

d=pd.read_csv('C:\mlfiles\songs.csv')
# print(d.isnull)

# d.isnull()
# print(d.isnull)
# print(d.info())
# print(d1_components.shape)
# d_components = d[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness','valence', 'tempo']]

# d2_components=[]\
# for i in d1_components:
#     d1_components['i']=d1_components['i'].apply(lambda k:k*10)
#     d2_components.append(d1_components['i'])
# # print(d2_components)
# print("hi arun")


d['danceability']=d['danceability'].apply(lambda k:k*10)
d['energy']=d['energy'].apply(lambda k:k*10)
d['loudness']=d['loudness'].apply(lambda k:k*10)
d['speechiness']=d['speechiness'].apply(lambda k:k*10)
d['acousticness']=d['acousticness'].apply(lambda k:k*10)
d['acousticness']=d['acousticness'].apply(lambda k:k*10)
d['instrumentalness']=d['instrumentalness'].apply(lambda k:k*10)
d['valence']=d['valence'].apply(lambda k:k*10)
d['tempo']=d['tempo'].apply(lambda k:k*10)
d['artist_info']=d['artist_info'].apply(lambda k:k*10)
d['artist_name']=d['artist_name'].apply(lambda k:k*10)
# d['artist_pop']=d['artist_pop'].apply(lambda k:k*10)
d['mood']=d['mood'].apply(lambda k:k*10)
d['language']=d['language'].apply(lambda k:k*10)
d_components = d[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness','valence', 'tempo']]

data_scaled = normalize(d_components)
data_scaled = pd.DataFrame(data_scaled, columns=d_components.columns)
# print(data_scaled)


d1_components = d[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness','valence', 'tempo','artist_info','artist_name','mood','language']]
print(d1_components)



# d1_components.danceability.astype(str)
# d1_components.energy.astype(str)
# d1_components.loudness.astype(str)
# d1_components.speechiness.astype(str)
# d1_components.acousticness.astype(str)
# d1_components.instrumentalness.astype(str)
# d1_components.valence.astype(str)
# d1_components.tempo.astype(str)


# d['']=d[''].apply(lambda k:k*10)
# for i in d1_components:
#     print(d1_components.i.astype(int))
# print(d.info())
# d2=[]
# for i in range(len(d1_components)):
#     d1_components['i']=d1_components['i'].apply(lambda k:k*10)
#     d2.append(d1_components['i'])
# ,'artist_info','artist_name','artist_pop','mood','language'
# print(d2)
# print(d1_components..astype(int))
# # print(d1_components)
# # print("vghj")
new_components=str(d1_components['danceability'].astype(int))+' '+str(d1_components['energy'].astype(int))+' '+str(d1_components['loudness'].astype(int))+' '+str(d1_components['speechiness'].astype(int))+' '+str(d1_components['acousticness'].astype(int))+' '+str(d1_components['instrumentalness'].astype(int))+' '+str(d1_components['valence'].astype(int))+' '+str(d1_components['tempo'].astype(int))+' '+d1_components['artist_info']+' '+d1_components['artist_name']+' '+d1_components['mood']+' '+d1_components['language']
print(new_components)
hello=TfidfVectorizer()
component_vectors=hello.fit_transform(new_components)
similarity_score=cosine_similarity(component_vectors)
print(similarity_score)
song_name=input("enter the song name:\n")
all_songs=d['name'].tolist()
# print(all_songs)
final_all_songs = []
for i in all_songs:
    if i not in final_all_songs:
            final_all_songs.append(i)
# print(final_all_songs)
near_song=difflib.get_close_matches(song_name,final_all_songs)
print("near_song is",near_song)
# from sklearn.preprocessing import normalize
# data_scaled = normalize(data)
# data_scaled = pd.DataFrame(data_scaled, columns=data.columns)