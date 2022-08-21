#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Dataset
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

#printing the data description
print(data.DESCR)

#Independent data - the columns are not dependent on each other
df = pd.DataFrame(data = data.data, columns= data.feature_names)
df.head()

#Adding a new column which will make the data dependent
df['Target'] = data.target
df.head()

#DATA PREPROCESSING
#Dropping the columns which are not required(latitude and longitude)
df = df.drop(columns=['Latitude','Longitude'])
df.head()

#Training model and checking accuracy

#Dividing the data into training and testing data

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
y = df.iloc[:,-1].values
df = df.drop(labels=['Target'], axis = 1)
X = df.iloc[:,:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = 42)

from sklearn import metrics
model = RandomForestRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy of prediction: ', r2_score(y_test, y_pred)*100)

#Plotting the data

plt.scatter(model.predict(X_train), y_train, color = 'green', label = 'Train Data')
plt.scatter(model.predict(X_test), y_test, color = 'yellow', label = 'Test Data')
#plt.hlines(y=0, xmin=0, xmax=50, linewidth = 2)
plt.legend(loc = 'lower right')
plt.title("House Price Predcition")
plt.show()
