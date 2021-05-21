import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
# print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head())

predict = "G3"  # final grade (also called label)

X = np.array(data.drop([predict], 1))  # all of my features or attributes
y = np.array(data[predict])  # what I want to predict

# Split into 4

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)  # fit the data to find the best fit line
acc = linear.score(x_test, y_test)  # return the value of accuracy of the model
print(acc)

print("Co: " , linear.coef_)
print("Intercept: " , linear.intercept_)
