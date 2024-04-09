# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:33:08 2024

@author: HP
"""

"""
Divide the diabetes data into train and test datasets and 
build a Random Forest and Decision Tree model with Outcome 
as the output variable. 
"""
"""
Business Objective:_
Minimize:- Need to minimize the deseases
Maximize:- Identify the root causes of the deseases
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("Diabetes.csv")

df.head(10)
df.tail()

# 5 number summary
df.describe()
"""
Number of times pregnant  ...   Age (years)
count                 768.000000  ...    768.000000
mean                    3.845052  ...     33.240885
std                     3.369578  ...     11.760232
min                     0.000000  ...     21.000000
25%                     1.000000  ...     24.000000
50%                     3.000000  ...     29.000000
75%                     6.000000  ...     41.000000
max                    17.000000  ...     81.000000

[8 rows x 8 columns]
"""
df.shape
#There are total 768 rows and 9 columns
df.columns
'''
[' Number of times pregnant', ' Plasma glucose concentration',
       ' Diastolic blood pressure', ' Triceps skin fold thickness',
       ' 2-Hour serum insulin', ' Body mass index',
       ' Diabetes pedigree function', ' Age (years)', ' Class variable']
'''

# check for null values
df.isnull()
# False
df.isnull().sum()
# 0 no null values

# Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

# Replace column names
# df.columns = df.columns.str.replace(' ', '_')
# df.columns

df.columns = [
    'Pregnant',
    'Glucose',
    'BloodPressure',
    'thickness',
    'insulin',
    'BMS',
    'Diabetes',
    'age',
    'class_variable'
]

# Now you can access the columns
print(df.columns)

# boxplot
# boxplot on Pregnant column
sns.boxplot(df.Pregnant)
# In Pregnant column 3 outliers 

sns.boxplot(df.age)
# In Income column many outliers

# boxplot on df column
sns.boxplot(df)
# There is outliers on all columns

# histplot - show distributions of datasets
sns.histplot(df['age'],kde=True)
# normally right skew and the distributed

sns.histplot(df['Pregnant'],kde=True)
# right skew and the distributed

sns.histplot(df,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# Data Preproccesing
df.dtypes
# Some columns in int, float data types and some Object

# Identify the duplicates
duplicate=df.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series is created
duplicate
# False
sum(duplicate)
# sum is 0.
# Normalize data 
# Normalize the data using norm function
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Apply the norm_fun to data 
df1=norm_fun(df.iloc[:,:8])

df['class_variable']
df1['class_variable']=df['class_variable']

df.isnull().sum()
df.dropna()
df.columns

# Converting into binary
lb=LabelEncoder()
df1["class_variable"]=lb.fit_transform(df1["class_variable"])

df1["class_variable"].unique()
df1['class_variable'].value_counts()
colnames=list(df1.columns)

predictors=colnames[:8]
target=colnames[8]

# Spliting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test=train_test_split(df1,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT

model=DT(criterion='entropy')
model.fit(train[predictors], train[target])
preds_test=model.predict(test[predictors])
preds_test
pd.crosstab(test[target], preds_test,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_test==test[target])

# Now let us check accuracy on training dataset
preds_train=model.predict(train[predictors])
pd.crosstab(train[target], preds_train,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_train==train[target])

# 100 % accuracy 
# Accuracy of train data > Accuracy test data i.e Overfit model
