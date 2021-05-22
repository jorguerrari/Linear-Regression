import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle


data = pd.read_csv("student-mat.csv", sep=";")
# print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head())

predict = "G3"  # final grade (also called label)

X = np.array(data.drop([predict], 1))  # all of my features or attributes
y = np.array(data[predict])  # what I want to predict
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)  # Split into 4

"""
## Create a loop to make sure I get the best model possible
best = 0
for _ in range (30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1) # Split into 4

    linear = linear_model.LinearRegression()  #comment out once the pickle file is saved

    linear.fit(x_train, y_train)  #fit the data to find the best fit line
    acc = linear.score(x_test, y_test)  #return the value of accuracy of the model
    print(acc)  #as an indicator to see how much each time

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:  # to save the model pickle
         pickle.dump(linear, f) """

# load the pickle file
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)

# now doing the predictions
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# Plot
p = "G2"  # use a variable to make it more dynamic
# look at correlation
pyplot.style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
