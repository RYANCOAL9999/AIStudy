import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score    

data = pd.read_csv("../dataSet/hp.csv")
data.info()

#deletes the three columns that contain string values and one column that contains missing values
data = data.drop(["airport", "waterbody", "bus_ter", "n_hos_beds"], axis = 1)
data.head(2)

#selects all rows for all columns except the last column or selects all rows for the independent variables but not the dependent variable
X = data.iloc[: , : -1].values
#select all rows for the last column which is the dependent variable
y = data.iloc[:, -1].values
# [[2.4000000e+01 ... 6.0335899e-02]]
print(X)
# [0 ... 1]
print(y)

#splits the data by allocating 20% of the data in the test set and 80% in the training set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
#creates an object of the LogisticRegression class and names is model
s = StandardScaler()
xTrain = s.fit_transform(X_train)
xTest = s.transform(X_test)
#creates an object of the LogisticRegression class and names is model
model = LogisticRegression()
#uses the created object to fit a Logisitc Regression model on the training data
model.fit(X_train, y_train)
#makes prediction on the out of sample data
y_pred = model.predict(X_test)
#outputs the actual values against the predicted values for easy visual inspection
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
# array([[39, 19], [12, 32]])
print(confusion_matrix(y_test, y_pred)) 
# (39+32)/(39+32+19+12) is equal too this
print(accuracy_score(y_test, y_pred))