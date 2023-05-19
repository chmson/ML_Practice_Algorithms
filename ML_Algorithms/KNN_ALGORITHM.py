import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
df = pd.read_csv("Classified Data",index_col=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
# StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],test_size=0.30)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=1, p=2,
#            weights='uniform')
pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
# print(confusion_matrix(y_test,pred))

error_rate = []


for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    print("for",i,pred_i)
    print("for",i,y_test.count())
    error_rate.append(np.mean(pred_i != y_test))
    print(error_rate)
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='red', linestyle='dashed', marker='o', markerfacecolor='green', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
# plt.show()

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

# print('WITH K=1')
# print('\n')
# print(confusion_matrix(y_test,pred))
# print('\n')
# print(classification_report(y_test,pred))

# NOW WITH K=23

knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
# print('WITH K=23')
# print('\n')
# print(confusion_matrix(y_test,pred))
# print('\n')
# print(classification_report(y_test,pred))