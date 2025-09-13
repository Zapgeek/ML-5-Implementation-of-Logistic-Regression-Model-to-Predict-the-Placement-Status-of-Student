# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, drop unnecessary columns, and encode categorical variables.
2. Define the features (X) and target variable (y).
3. Split the data into training and testing sets.
4. Train the logistic regression model, make predictions, and evaluate using accuracy and other

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Aman Singh
RegisterNumber: 212224040020
*/
```
```python

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no", "salary"], axis=1) 
data1.head()
data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"]) 
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"]) 
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression (solver="liblinear") 
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)
y_pred
lr.score(x_test,y_test)

from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred) 
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,0,1,1,90,1,0,85,1,85]])

```

## Output:
<img width="1179" alt="Screenshot 2025-05-13 at 9 32 51 AM" src="https://github.com/user-attachments/assets/53c088ba-1520-4fa0-bdba-62041fd71f0c" />

<img width="1085" alt="Screenshot 2025-05-13 at 9 33 03 AM" src="https://github.com/user-attachments/assets/ac8b5397-ed94-472a-9e85-5a9b7a75c19a" />

<img width="309" alt="Screenshot 2025-05-13 at 9 33 13 AM" src="https://github.com/user-attachments/assets/4dd7bad8-e880-4ad3-b7e3-0e524989d2c3" />

<img width="1024" alt="Screenshot 2025-05-13 at 9 33 25 AM" src="https://github.com/user-attachments/assets/ffc098b4-9308-424e-ab0e-0c0eeefaa636" />

<img width="882" alt="Screenshot 2025-05-13 at 9 33 39 AM" src="https://github.com/user-attachments/assets/b5d6c4a3-d8ad-4c13-87e3-535815699eb3" />

<img width="230" alt="Screenshot 2025-05-13 at 9 33 51 AM" src="https://github.com/user-attachments/assets/890de6c4-c441-4e05-a383-42880882345e" />

<img width="707" alt="Screenshot 2025-05-13 at 9 34 03 AM" src="https://github.com/user-attachments/assets/e7deea9f-0b3c-4129-984a-976e95546f8b" />

<img width="541" alt="Screenshot 2025-05-13 at 9 34 16 AM" src="https://github.com/user-attachments/assets/86f667c3-3b9e-412f-923d-9c22c75545ff" />

<img width="644" alt="Screenshot 2025-05-13 at 9 34 34 AM" src="https://github.com/user-attachments/assets/3c689f68-fcb3-4d13-bd16-b1f68e2d430b" />

<img width="528" alt="Screenshot 2025-05-13 at 9 34 45 AM" src="https://github.com/user-attachments/assets/e39b5b1b-2f31-4f97-8800-a1ab8c2a0d9d" />

<img width="928" alt="Screenshot 2025-05-13 at 9 34 59 AM" src="https://github.com/user-attachments/assets/00e9ae52-b265-42df-8c76-4cc840557195" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
