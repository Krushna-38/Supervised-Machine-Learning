# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:32:23 2024

@author: HP
"""

"""
1.	A cloth manufacturing company is interested to know
 about the different attributes contributing to high 
 sales. Build a decision tree & random forest model with
 Sales as target variable (first convert it into categorical variable).
"""
"""
Business Objective
Minimize:- Need to minimize the defects present in cloths
Maximize:- Quality of cloths, so that sale will be increased
"""

import pandas as pd
import numpy as np
cloth = pd.read_csv("C:/Dataset/Company_Data.csv")

#Let's perform the EDA
cloth.shape
# (rows=400, columns=11) 
cloth.columns
"""
Index(['Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'ShelveLoc', 'Age', 'Education', 'Urban', 'US'],
      dtype='object')
"""
cloth.size
#4400
cloth.describe()
cloth.head()
#To check null values in the Datasets
cloth.isnull()
#False. 
cloth.isnull().sum()
# 0.It means there is no any null value present in the dataset
#Normalization
def norm(i):
    x = (i-i.min())/(i.max()-i.min())
    return x
from sklearn.preprocessing import LabelEncoder

# Assuming df is your DataFrame and 'ShelveLoc' is the column you want to encode
le = LabelEncoder()
cloth['ShelveLoc'] = le.fit_transform(cloth['ShelveLoc'])
cloth['Urban'] = le.fit_transform(cloth['Urban'])
cloth['US'] = le.fit_transform(cloth['US'])

cloth_norm = norm(cloth.iloc[:,0:16])

# Now, 'ShelveLoc' column will have numerical values 
#corresponding to 'Good', 'Bad', 'Medium'


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


le_ShelveLoc = LabelEncoder()
le_Urban = LabelEncoder()
le_US = LabelEncoder()
cloth['ShelveLoc'] = le_ShelveLoc.fit_transform(cloth['ShelveLoc'])
cloth['Urban'] = le_Urban.fit_transform(cloth['Urban'])
cloth['US'] = le_US.fit_transform(cloth['US'])

# Separate features and target variable
from sklearn.tree import DecisionTreeRegressor
X = cloth.drop("Sales", axis='columns')
y = cloth["Sales"]

# Decision Tree model
model = DecisionTreeRegressor()
model.fit(X, y)

# Example predictions
# Assuming ShelveLoc=2, Urban=1, US=0 for the first example
prediction_1 = model.predict([[115, 95, 5, 110, 117, 26, 20, 0, 1, 1]])

# Assuming ShelveLoc=2, Urban=1, US=1 for the second example
prediction_2 = model.predict([[165, 70, 8, 200, 130, 33, 20, 1, 0, 1]])

print("Prediction 1:", prediction_1)
print("Prediction 2:", prediction_2)