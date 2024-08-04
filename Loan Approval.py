import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

location= r'C:\Users\mayan\Downloads\loan_small.csv'
dataset=pd.read_csv(location)

cleandata = dataset.dropna(subset=['Loan_Status'])
df=cleandata.copy()

#replace the categorial value with mode
cols = ['Gender','Area','Loan_Status']
df[cols] = df[cols].fillna(df.mode().iloc[0])

#replace the categorial value with mean
cols2 = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
df[cols2] = df[cols2].fillna(df[cols2].mean())

#drop irrevelent columns
df = df.drop(['Loan_ID'],axis=1)

#create dummy variable or Hoy encoding
df= pd.get_dummies(df,drop_first=True)
# print(df)

#split the data verticaly int x and y
X = df.drop('Loan_Status_Y', axis=1)
y = df['Loan_Status_Y']

# Split the dataset by rows
# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Training complete.")
print("Predicted values for the test set:", y_pred)
print("Actual values for the test set:", y_test.values)