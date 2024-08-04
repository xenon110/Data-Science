import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

location = r'C:\Users\mayan\Downloads\01Students.csv'
dataset = pd.read_csv(location)

X = dataset[['Hours']]
Y = dataset[['Marks']]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=1234)

model = LinearRegression()
model.fit(X_train,Y_train)


# R square value of the test
slr_score = model.score(X_test,Y_test)

# Predict the value of y according to data its not necessay but
# will help in finding the RMSE
y_predict = model.predict(X_test)

# cofficient of regresion line
slr_coefficient = model.coef_
slr_intersept = model.intercept_

print(f"intersept value is: {slr_intersept}")
print(f"coeficien value is: {slr_coefficient}")

#Equatioin of the line will be
# y = 34.27 + 5.02 * x

# How much our model has made the error
# RMSE - Root Mean Square E rror
from sklearn.metrics import mean_squared_error
import math

slr_rmse = math.sqrt(mean_squared_error(Y_test,y_predict))
print(f"RMSE VALUE or Error value {slr_rmse}")

# PLTING IN GRAPH
import matplotlib.pyplot as plt

plt.scatter(X_test,Y_test)

# Trending of the prediction
plt.plot(X_test,y_predict)
plt.show()

#>>>>>>>>>>>>>>>>>>>>>>>     BY USER INPUT    >>>>>>>>>>>>>>
#User input
user_input = float(input("Enter the number of hours you study:"))
predicted_score = model.predict([[user_input]])

#Printing the output
print(f"predicted Exam Score is : {predicted_score[0]}")


