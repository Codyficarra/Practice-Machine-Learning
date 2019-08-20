import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plot
import pickle
from matplotlib import style

#reading in the data
data = pd.read_csv("student-mat.csv", sep=";")

#the data we want to predict
pred = data[["G3"]]

#situating the data we are using for prediction
data = data[["G1", "G2", "studytime", "failures", "absences"]]

x = np.array(data)

y = np.array(pred)

#best prediction score
best = 0

for i in range(200):
    #creating the training and test data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)

    #creating our linear model
    linear = linear_model.LinearRegression()

    linear.fit(x_train,y_train)

    #getting the accuracy of our prediction
    accuracy = linear.score(x_test, y_test)

    if accuracy > best:
        if i > 0:
            print("Accuracy was improved to:",accuracy, " on attempt #:", i)
        best = accuracy
        b_x_train = x_train
        b_x_test = x_test
        b_y_train = y_train
        b_y_test = y_test
        with open("grade_prediction_model.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("grade_prediction_model.pickle", "rb")

linear = pickle.load(pickle_in)

accuracy = linear.score(b_x_test,b_y_test)

print("\n")

print("Coefficient:", linear.coef_)

print("Intercept:", linear.intercept_, "\n")

prediction = linear.predict(b_x_test)

ave_err = []

print("\n")

for x in range(len(prediction)):
    print("best_prediction of x:", prediction[x], "  Actual answer:", b_y_test[x], "  Error:", abs(prediction[x] - b_y_test[x]))
    ave_err.append(abs(prediction[x] - b_y_test[x]))

print("\n")
print("Accuracy:", accuracy)
print("Average Error of best model:", np.mean(ave_err))


style.use("ggplot")

p = 'G2'

plot.scatter(data[p],pred)
plot.xlabel(p)
plot.ylabel("Final Grade")
plot.show()