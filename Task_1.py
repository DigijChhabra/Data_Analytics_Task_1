#Task 1
import pandas as pd
data = pd.read_csv("F:\\Digij\\Internshp TSF\\Internship\\data1.csv")
data.head()
print("Data Imported Successfully")

import matplotlib.pyplot as plt
x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
plt.scatter(x='Hours', y='Scores',data=data)
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print("Data has been trained successfully")

line = regressor.coef_*x+regressor.intercept_
plt.scatter(x, y)
plt.plot(x, line,color = "red")
plt.show()

pred = regressor.predict(x_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
print(df)

import numpy as np
hours = np.array(9.25).reshape(-1,1)
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))

from sklearn import metrics
print('Mean Absolute Error = ', metrics.mean_absolute_error(y_test, pred))
