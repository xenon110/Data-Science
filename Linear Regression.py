'''
problem statement is
    Predict a student final exam score based on the number of
        hour they study
'''
# STEP 1 import every usefull  moduel

import streamlit as st
import numpy as np     #when we deal with 2 numeric value then we use this
import pandas as pd    # when we deal with the tabular value
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# TAKE DATA

sharadata = {'Hour_study':[0,1,2,3,4,5,6,7,8,9],
             'Exam_score':[0,20,35,50,65,75,79,80,90,94]}

# Step 3
# This is data frame of data
df = pd.DataFrame(sharadata)

#Step 4
# feature extraction (feature=column)
X = df[['Hour_study']]
Y = df[['Exam_score']]

# Note ab hm datatrain krege jo ki 80 20 hoga mtlb jo data hmlog ne liya hai
# usko 80% train krege or 20 % se validate krege  sb log apne mn se lete hai but kvi v 50% 50% mat le lena

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# AB training testing to bta diya ab kis model pe krna hai ab usko btana hai so
# we are doing on linear regresion

#Step 6
model = LinearRegression()

#step 7
model.fit(X_train,Y_train)

#User input
user_input = float(input("Enter the number of hours you study:"))
predicted_score = model.predict([[user_input]])

#Printing the output
print(f"predicted Exam Score is : {predicted_score[0]}")
