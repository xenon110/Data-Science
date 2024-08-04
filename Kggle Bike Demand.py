# ------------------------------------------
#  Bike sharing Demand predication project for the hourly dataset
#-------------------------------------------------------
#--------------------------------------------------
#   Important Libraries
#--------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
#---------------------- STEP 1 ------------------------

location = r"C:\Users\mayan\Downloads\hour.csv"
bike = pd.read_csv(location)

#--------------------   STEP 2 ---------------------------
bike_prep = bike.copy()
bike_prep = bike_prep.drop(['index', 'date', 'casual', 'registered'], axis = 1)
'''
# check is their is any missing value
bike_prep.isnull().sum()
bike_prep.hist(rwidth=0.9)
plt.tight_layout()
# plt.show()
'''
#------------------------- STEP 3 ---------------------------
#  DATA VISULIZATION
#Visulise the continious features vs demand
#------------------------------------------------------------
'''plt.subplot(2,2,1)
plt.title('Tempreture Vs Demand ')
plt.scatter(bike_prep['temp'],bike_prep['demand'],s=3)

plt.subplot(2,2,1)
plt.title('Tempreture Vs Demand ')
plt.scatter(bike_prep['temp'],bike_prep['demand'],s=3,c='g')

plt.subplot(2,2,2)
plt.title('atemp Vs Demand ')
plt.scatter(bike_prep['atemp'],bike_prep['demand'],s=3,c='b')

plt.subplot(2,2,3)
plt.title('Humidity Vs Demand ')
plt.scatter(bike_prep['humidity'],bike_prep['demand'],s=3,c='m')

plt.subplot(2,2,4)
plt.title('Windspeed Vs Demand ')
plt.scatter(bike_prep['windspeed'],bike_prep['demand'],s=3,c='c')

plt.tight_layout()
# plt.show()
'''
#plot the categorial features vs demand
# create  a list of unique season value
'''
colours = ['g','r','b','m']

plt.subplot(3,3,1)
cat_list = bike_prep['season'].unique()
cat_average = bike_prep.groupby('season').mean()['demand']
plt.bar(cat_list,cat_average,color = colours)
plt.title('season Vs demand')

plt.subplot(3,3,2)
cat_list = bike_prep['year'].unique()
cat_average = bike_prep.groupby('year').mean()['demand']
plt.bar(cat_list,cat_average,color = colours)
plt.title('Year Vs Demand')

plt.subplot(3,3,3)
cat_list = bike_prep['month'].unique()
cat_average = bike_prep.groupby('month').mean()['demand']
plt.bar(cat_list,cat_average,color = colours)
plt.title('Month Vs Demand')

plt.subplot(3,3,4)
cat_list = bike_prep['hour'].unique()
cat_average = bike_prep.groupby('hour').mean()['demand']
plt.bar(cat_list,cat_average,color = colours)
plt.title('Hour Vs Demand')

plt.subplot(3,3,5)
cat_list = bike_prep['workingday'].unique()
cat_average = bike_prep.groupby('workingday').mean()['demand']
plt.bar(cat_list,cat_average,color = colours)
plt.title('Working day Vs Demand')

plt.subplot(3,3,6)
cat_list = bike_prep['holiday'].unique()
cat_average = bike_prep.groupby('holiday').mean()['demand']
plt.bar(cat_list,cat_average,color = colours)
plt.title('Holiday Vs Demand')

plt.subplot(3,3,7)
cat_list = bike_prep['weekday'].unique()
cat_average = bike_prep.groupby('weekday').mean()['demand']
plt.bar(cat_list,cat_average,color = colours)
plt.title('Weekday Vs Demand')

plt.subplot(3,3,8)
cat_list = bike_prep['weather'].unique()
cat_average = bike_prep.groupby('weather').mean()['demand']
plt.bar(cat_list,cat_average,color = colours)
plt.title('Weather Vs Demand')

plt.tight_layout()
plt.show()
'''
#-------------------------------------------
# Check the outlier
#--------------------------------------------

detail = bike_prep['demand'].describe()
quantiles = bike_prep['demand'].quantile([0.05,0.10,0.15,0.90,0.95,0.99])
# print(quantiles)

#-----------------------------------------------
# Step 4 - Check multiple linear regeressio assumption
#-----------------------------------------------
correlation =bike_prep[['temp','atemp','humidity','windspeed','demand']].corr()
# print(correlation)
#NOTE -> atem and windspeed their correlatioin cofficient are more so drop it

#---------------------------------------------------------
# Step 5 Droping the unwanted column
#---------------------------------------------------------

bike_prep = bike_prep.drop(['atemp','weekday','year','workingday','windspeed'],axis=1)
# print(bike_prep)
# Check autocorellatioin in demand using acorr
df = pd.to_numeric(bike_prep['demand'],downcast='float')
plt.acorr(df,maxlags=12)
# plt.show()
#------------------------------------------------------
#Step 6  create /modify new features
#------------------------------------------------------
# Log normalize the feature demand
'''
df1 = bike_prep['demand']
df2 = np.log(df1)

plt.figure()
df1.hist(rwidth=0.9,bins=20)
plt.title('Normal distribution of graph')

plt.figure()
df2.hist(rwidth=0.9,bins=20)
plt.title('Log distribution of graph')
plt.show()
'''
# Convert the normal value of deamnd to the log value
bike_prep['demand'] = np.log(bike_prep['demand'])

# Auto colrelation in the demand column
t_1 = bike_prep['demand'].shift(+1).to_frame()
t_1.columns = ['t-1']

t_2 = bike_prep['demand'].shift(+2).to_frame()
t_2.columns = ['t-2']

t_3 = bike_prep['demand'].shift(+3).to_frame()
t_3.columns = ['t-3']

bike_prep_log = pd.concat([bike_prep,t_1,t_2,t_3],axis=1)
bike_prep_log = bike_prep_log.dropna()

#------------------------------------------------------
#Step 7 create Dummy variable and drop first
#        to avoid dummy variable trap using get_dumminies
#------------------------------------------------------
# - season holiday weather hour month

#Changing the datatype to category
bike_prep_log['season'] = bike_prep_log['season'].astype('category')
bike_prep_log['holiday'] = bike_prep_log['holiday'].astype('category')
bike_prep_log['weather'] = bike_prep_log['weather'].astype('category')
bike_prep_log['hour'] = bike_prep_log['hour'].astype('category')
bike_prep_log['month'] = bike_prep_log['month'].astype('category')

bike_prep_log =pd.get_dummies(bike_prep_log,drop_first=True)

#Split the X nd Y data set into training and testing set

Y = bike_prep_log[['demand']]
X = bike_prep_log.drop(['demand'],axis=1)

# Create 70% of the data to train
tr_size = 0.7*len(X)
tr_size = int(tr_size)

X_train = X.values[0:tr_size]
X_test = X.values[tr_size:len(X)]

Y_train = Y.values[0:tr_size]
Y_test = Y.values[tr_size:len(Y)]

# Linear Regression
from sklearn.linear_model import  LinearRegression

model = LinearRegression()
model.fit(X_train,Y_train)
r2_train = model.score(X_train,Y_train)
r2_test = model.score(X_test,Y_test)

# create y prediction
y_predict = model.predict(X_test)

from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(Y_test,y_predict))
print(f"Error value {rmse}") # average magnitude of the errors between predicted and actual values.
# print(f"r2 train : {r2_train}")
# print(f"r2 test : {r2_test}")

# Calculate the Rmsle
Y_test_e = []
Y_predict_e = []

for i in range(0,len(Y_test)):
    Y_test_e.append(math.exp(Y_test[i]))
    Y_predict_e.append(math.exp(y_predict[i]))

#Do the sum of the logs and square
log_sq_sum = 0.0
for i in range(0,len(Y_test_e)):
    log_a = math.log(Y_test_e[i] + 1)
    log_p = math.log(Y_predict_e[i] + 1)
    log_diff = (log_p - log_a)**2
    log_sq_sum = log_sq_sum + log_diff
rmsle = math.sqrt(log_sq_sum/len(Y_test))
print(rmsle)
