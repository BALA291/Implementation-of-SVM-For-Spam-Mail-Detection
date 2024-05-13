# EX-09 Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: BALAMURUGAN B
RegisterNumber:  212222230016
*/
```
```python
import chardet 
file="CSVs/spam.csv"
with open(file,'rb')as rawdata: 
    result = chardet.detect(rawdata.read(100000)) 
result
import pandas as pd 
data=pd.read_csv("CSVs/spam.csv",encoding="'Windows-1252") 
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values 
y=data["v2"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()
x_train=cv.fit_transform(x_train) 
x_test=cv.transform(x_test)
from sklearn.svm import SVC 
svc=SVC() 
svc.fit(x_train,y_train) 
y_pred=svc.predict(x_test) 
y_pred
from sklearn import metrics 
accuracy=metrics.accuracy_score(y_test,y )  
accuracy
```

## Output:
##### df.head():
![image](https://github.com/BALA291/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120717501/b01c2e41-4beb-43e6-8896-674ba29b670d)

#### df.info():
![image](https://github.com/BALA291/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120717501/6b2861e9-f88c-4602-9db5-5b6b1849ddde)

#### df.null():
![image](https://github.com/BALA291/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120717501/c44721cf-49ce-4af3-81c4-d4bb4ec6d71c)

#### y_pred:
![image](https://github.com/BALA291/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120717501/35c49c2a-e369-4e18-a528-511457da1fcd)

#### Accuracy:
![image](https://github.com/BALA291/Implementation-of-SVM-For-Spam-Mail-Detection/assets/120717501/dcd90555-cf89-4d9a-9631-17eb19e21e8c)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
