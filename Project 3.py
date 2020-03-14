import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns 

from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm    		


data = pd.read_csv('C:/Users/patel/AppData/Local/Programs/Python/Python38/Programs/Assignment Programs/Project/Data Set/train.csv')

print(data.head(10))

data.species = pd.Categorical(data.species)
data['species'] = data.species.cat.codes

print(data.head(10))


random.seed(4)
pred_columns = data[:]

prediction_var = pred_columns.columns


train, test = train_test_split(data, test_size = 0.2,random_state=27)

print("\n \n The values from train set by stratifying them and dividing it into 80:20 ratio are foloowing :  ")

print("\n \n The shape of the Train Data is       :                 ",train.shape)
print("\n \n The shape of the Test Data is         :                ",test.shape)


train_X = train[prediction_var]
train_y= train['species']

test_X= test[prediction_var] 
test_y =test['species'] 


#RandomForest classifier

model=RandomForestClassifier(n_estimators=100,random_state=27)
model.fit(train_X,train_y)
prediction=model.predict(test_X)

print("\n \n The accuracy score for given data set using the RANDOM FORSET ALGORITHM is                 :                    ",metrics.accuracy_score(prediction,test_y))


#Decision Tree

model = tree.DecisionTreeClassifier(random_state=27)
model.fit(train_X,train_y)
prediction=model.predict(test_X)


print("\n \n The accuracy score for given data set using the DECISION TREE ALGORITHM is                 :                    ",metrics.accuracy_score(prediction,test_y)) 

'''

# Support Vector Machine

model = svm.SVC(kernel='linear',random_state=27)
model.fit(train_X,train_y)

predicted= model.predict(test_X)
print(" \n \n The accuracy score for given data set using the SVM ALGORITHM  is                 :                    ",accuracy_score(test_y, predicted),"\n \n")

'''

# Navie Bayes

gnb = GaussianNB(priors=None, var_smoothing=1e-09)

y_pred_gnb = gnb.fit(train_X, train_y,sample_weight=None)
target_pred = y_pred_gnb.predict(test_X)


cnf_matrix_gnb = confusion_matrix(test_y,target_pred)

print("\n \n The Confusion Matrix is     :         \n \n ",cnf_matrix_gnb)

print("\n \n The accuracy score for given data set using the  Navie Bayes Theorm is             :                  ",metrics.accuracy_score(target_pred,test_y),"\n \n")



print("\n \n Out of all the 4 Algorithms the SUPPORT VECTOR MACHINE Algorithm gives the best accuracy for the given data set.      \n \n")
