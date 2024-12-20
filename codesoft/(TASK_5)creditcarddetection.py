import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

data=pd.read_csv(r"C:/Users/basud/Downloads/creditcard.csv/creditcard.csv")
data
data.shape
data.columns
data.info()
data.isnull().sum()
data=data.drop('Time',axis=1)
data
data['Class'].unique()
#cheaking the duplicates value
data.duplicated().any()
data[data.duplicated()]
data.drop_duplicates(keep='first',inplace=True)
data.duplicated().sum()
data.shape
data.describe()
len(data[data['Class']==0])
len(data[data['Class']==1])
data['Class'].value_counts()
sns.countplot(x=data['Class'],hue=data['Class'])
x=data.drop('Class',axis=1)
y=data['Class']
#appropriate model fitting
import statsmodels.api as sm 
x1=sm.add_constant(x)
logit_model=sm.Logit(y,x1)
result=logit_model.fit()
print(result.summary())
import scipy
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
x=data.drop('Class',axis=1)
y=data['Class'].ravel()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=22)
len(x_train),len(x_test),len(y_test),len(y_train)
Lreg=LogisticRegression(random_state=22)
Lreg.fit(x_train,y_train.ravel())
y_predict=Lreg.predict(x_test).ravel()
y_predict

data.Class 
Lreg.score(x_test,y_test)
## create confusion matrix
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_predict)
score
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_predict)
tn,fp,fn,tp=confusion_matrix(y_test,y_predict).ravel()
print("True Negative",tn)
print("False Positive",fp)
print("False Negative",fn)
print("True Positive",tp)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))
ACCURACY= (tp+tn)/(tp+tn+fp+fn)
print("Accuracy {:0.2f}".format(ACCURACY))
PRECISION=(tp)/(tp+fp)
print("Precision {:0.2f}".format(PRECISION))
RECALL=tp/(tp+fn)
print("Sensitivity {:0.2f}".format(RECALL))
specifity=(tn)/(tn+fp)
print("specificity {:0.2f}".format(specifity))

