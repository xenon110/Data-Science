
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

location = r"C:\Users\mayan\Downloads\02Students.csv"
dataset = pd.read_csv(location)

df = dataset.copy()
# Split the variable
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1234)

model = LinearRegression()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)
# y =a + b1x1 + b2x2
# y =1.31 + 4.67*hour + 5.1*sleephour

mlr_score = model.score(x_test,y_test)
mlr_cofe = model.coef_
mlr_intersept = model.intercept_
# How much our model has made the error
# RMSE - Root Mean Square E rror
from sklearn.metrics import mean_squared_error
import math

mlr_rmse = math.sqrt(mean_squared_error(y_test,y_predict))
print(f"Error value {mlr_rmse}") # average magnitude of the errors between predicted and actual values.
print(f"cofficient : {mlr_cofe}")
print(f"Intersept : {mlr_intersept}")
print(f"Mlr Score : {mlr_score}")  # This will tell how much your data is accurate







