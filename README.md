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
Developed by: ADHITHYARAM D
RegisterNumber:  212222230008
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
<img src = "https://github.com/Adhithyaram29D/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393540/f59f30db-3832-4701-b5ec-0381c9fc0af2" width="300">

#### df.info():
<img src = "https://github.com/Adhithyaram29D/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393540/43c833c1-9f69-4fc0-8566-4a92e2407d29" width="200">

#### df.null():
<img src = "https://github.com/Adhithyaram29D/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393540/41fd497f-32b9-4c1e-9f83-aac4252f4804" width="100">

#### y_pred:
<img src = "https://github.com/Adhithyaram29D/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393540/039c0e79-4bfc-474a-83e1-445165e67962" width="300">

#### Accuracy:
<img src = "https://github.com/Adhithyaram29D/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393540/888b825d-0e49-4a1e-a26f-babce1862d07" width="100">

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
