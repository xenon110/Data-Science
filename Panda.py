'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

 b
dataset=pd.read_csv(location)
# print(dataset)

subset=dataset.iloc[0:3,1:3]
subset1=dataset[dataset['Gender'] == 'Male']
subset2=dataset[dataset['LoanAmount'] >=100]
subset3=dataset[['Gender','LoanAmount']]
'''
'''
print(subset3)
print(dataset.head)
print(dataset.shape)

# Find out the missing value from the dataset
print(dataset.isnull().sum(axis=0))  --> Column
print(dataset.isnull().sum(axis=1))  --> ROW

#Finding the full dataset info where the value is null
print(dataset[dataset['ApplicantIncome'].isnull()])

print(dataset.isnull().sum(axis=0))
#this will delet all the record
print(dataset.dropna())

clean = dataset.dropna(subset=['Area'])
print(clean)
'''
# dt=dataset.copy()
#Replace the missing value with the Mode
'''
cols = ['Gender','Area','Loan_Status']
dt[cols] = dt[cols].fillna(dt[cols].mode().iloc[0])
print(dt[cols])
print(dt.isnull().sum(axis=0))
'''
# Replace the missing value with the  Mean value
'''
col2 = ['ApplicantIncome','LoanAmount']
dt[col2] = dt[col2].fillna(dt[col2].mean())
print(dt[col2])
print(dt.isnull().sum(axis=0))
'''
#Finding avg to chek that it is correct for the above one or not
'''
sums = dt['ApplicantIncome'].sum()
coun = dt['ApplicantIncome'].count()
print(sums/coun)
'''
'''
cols = ['Gender','Area','Loan_Status']
print(dt.dtypes)

dt[cols] = dt[cols].astype('category')
print(dt[cols].dtypes)

'''

# We are filling the categorial data to deal with the math
# First filll all the Nan data
'''
cols = ['Gender','Area','Loan_Status']
dt[cols] = dt[cols].fillna(dt[cols].mode().iloc[0])

col2 = ['Gender', 'Area', 'Loan_Status']

#Change the data type
dt[col2]=dt[col2].astype('category')
print(dt.dtypes)
for columns in col2:
    dt[columns] = dt[columns].cat.codes
# print(dt)

#Hot encoding
df2=dt.drop(['Loan_ID'],axis=1)
# Perform one-hot encoding
df2 = pd.get_dummies(df2)
print(df2)
'''

'''
# I have writen this code
    first i have clean the na value from the data
    second i have changed the data type from of the gender 

clean = dt.dropna()
col = ['Gender']
clean[col] = clean[col].astype('category')
for cols in col:
    clean[cols] = clean[cols].cat.codes
print(clean)
# data_to_scale = clean.iloc[:,1:5]

print(clean)
'''


'''
#       DATA NORMALIZATION

clean = dt.dropna()
#Extract the 3 numerical column
data_to_scale = clean.iloc[:,2:5]
# DATA NORMALIZATION USING standar scaler
from sklearn.preprocessing import StandardScaler
scaler_ = StandardScaler()
ss_scaler = scaler_.fit_transform(data_to_scale)
print(ss_scaler)

#       DATA NORMALIZATION USING MIN MAX
from sklearn.preprocessing import minmax_scale
mm_scaler = minmax_scale(data_to_scale)
print(mm_scaler)

'''













































#:::::::::::VIsulazing through GRAPH:::::::::
'''
x=dataset['ApplicantIncome']
y=dataset['LoanAmount']
plt.scatter(x,y)
plt.title('Applicant Income vs Loan Amount')
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.grid(True)
plt.show()
'''

#visulazing the value of the null in each of the column name with the bar graph
'''
x=dataset.isnull().sum(axis=0)
y = dataset.columns.tolist()

plt.figure(figsize=(12, 6))

#     BAR GRAPH
plt.subplot(1,2,1)
plt.bar(y,x)
plt.title('Number of Null Values per Column')
plt.xlabel('Columns')
plt.ylabel('Number of Null Values')
plt.xticks(rotation=45)
plt.grid(True)


#        Pie chart
plt.subplot(1,2,2)
plt.pie(x,labels=y,autopct='%2f%%')
plt.title('Number of Null ')
plt.tight_layout()

YAHA PLT.SHOW LIKH LENA

'''