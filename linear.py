# Importing Required Liberaries
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# Loading dataset
insurance_data = pd.read_csv("insurance.csv")

# Data Observations.
insurance_data
insurance_data.head()

# DATA PREPROCESSING 
#______________________________________________________________________
# Scatter Plot to get relation between bmi and charges and smoker as hue.
sns.scatterplot(x = insurance_data["bmi"], y = insurance_data["charges"], hue = insurance_data["smoker"])

# Creating input and output variables.

X = insurance_data.drop(columns=["charges", "region"])# Dropping Charges to create it as output. 
Y = insurance_data["charges"] # Output

X.head()
# Mapping male and female values to 0 and 1
X["sex"] = X ["sex"].map({"female":1 , "male":0})
X.head()
# Mapping Smoker Yes and No values to 0 and 1
X["smoker"] = X["smoker"].map({"yes": 1, "no": 0})
X.head()
#____________________________________________________________________________

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2,random_state=42)
X_train.head()
X_test.head()

# Training Model (Linear Regression)
model = LinearRegression()
model.fit(X_train, Y_train)

# Predicting Values

Y_pred = model.predict(X_test)
print("Predcted values:", Y_pred)# Predicted Values

# Actual Values 
Y_test

# Model Evaluation 
# R-squared
r2 = r2_score(Y_test, Y_pred) # r2_score is 0.78

# Calculating adjusted R-Squared
n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1-((1-r2)*(n-1)/ (n-p-1))
adjusted_r2# 0.7769