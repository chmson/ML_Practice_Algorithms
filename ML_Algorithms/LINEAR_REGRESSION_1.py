
from sklearn import linear_model
import pandas as pd
from matplotlib import style
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np

#print(one)
#plt.scatter(one['year'],one['per capita income (US$)'])
#plt.show()
# def best_fit_line(xs,ys):
#      slope = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * ys)))
#      y_intercept = mean(ys) - slope * mean(xs)
#      return slope, y_intercept
one=pd.read_csv('C:\\mlfiles\\income.csv')
# scaler = MinMaxScaler()
# scaler.fit(one[['year']])
# one['year'] = scaler.transform(one[['year']])
# scaler.fit(one[['per capita income (US$)']])
# one['per capita income (US$)'] = scaler.transform(one[['per capita income (US$)']])
year_list=one['year'].tolist()
income_list=one['per capita income (US$)'].tolist()
xs=np.array(year_list)
ys=np.array(income_list)
# slope, y_intercept = best_fit_line(xs,ys)
# regression_line = [(slope * x) + y_intercept for x in xs]
per_income = linear_model.LinearRegression()

# Train the model using the training sets
per_income.fit(xs.reshape(-1,1),ys)

# get the regression line using the model
regression_line = per_income.predict(xs.reshape(-1,1))

# Making predictions
predict_year = 1969
predict_income = per_income.predict(np.array([[predict_year]]))
print(predict_income)
# # Making predictions
# predict_year = 1978
# predict_income = (slope * predict_year) + y_intercept

# reg=linear_model.LinearRegression()
# reg.fit(xs.reshape(-1,1),ys)
# print(one.predict(2020))
# print(predict_income)
style.use('seaborn')
plt.scatter(xs,ys,label='Data Points', alpha=0.6,color='green',s=75)
plt.scatter(predict_year,predict_income,label='income_prediction',color='red',s=100)
plt.plot(xs,regression_line,label='Best Fit Line', color='orange',linewidth=4)
plt.title('Height and Weight linear regression')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend()
plt.show()


word='Perfect'
if word[-1]=='t':
    word=word[:-1]
    print(word)

