import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
marketing_df = pd.read_csv('https://raw.githubusercontent.com/srivatsan88/YouTubeLI/master/dataset/marketing_cva_f.csv')
print(marketing_df)
marketing_df=marketing_df.drop(['Customer','Vehicle_Class','avg_vehicle_age','months_last_claim','Total_Claim_Amount'],axis=1)
print(marketing_df)
mark_array=marketing_df.values
mark_array[:, 1] = mark_array[:, 1].astype(float)
mark_array[:, 3] = mark_array[:, 3].astype(float)
mark_array[:, 5] = mark_array[:, 5].astype(float)
mark_array[:, 6] = mark_array[:, 6].astype(float)
print(mark_array)
kproto = KPrototypes(n_clusters=3, verbose=2,max_iter=20)
clusters = kproto.fit_predict(mark_array, categorical=[0, 2, 4])
print(kproto.cluster_centroids_)
cluster_dict=[]
for c in clusters:
    cluster_dict.append(c)
print(cluster_dict)
marketing_df['cluster']=cluster_dict
print(marketing_df)
marketing_df[marketing_df['cluster']== 0].head(20)
marketing_df[marketing_df['cluster']== 1].head(20)
marketing_df[marketing_df['cluster']== 2].head(20)