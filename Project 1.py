import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
from memory_profiler import profile
from sklearn.tree import DecisionTreeClassifier


def something_to_profile():

    data = pd.read_csv("C:/Users/patel/AppData/Local/Programs/Python/Python38/Programs/Assignment Programs/Project/Data Set/OnlineNewsPopularity.csv")

    data.url = pd.Categorical(data.url)
    data['url'] = data.url.cat.codes

    corr=data.corr()
    
    sns.heatmap(corr,square=True,cmap="BuGn")
    plt.show()


    random.seed(4)
    pred_columns = data.iloc[:,1:62]

    pred_columns.drop([ 'timedelta'],axis=1,inplace=True)
    pred_columns.drop(['num_self_hrefs'],axis=1,inplace=True)
    pred_columns.drop(['kw_min_min'],axis=1,inplace=True)
    pred_columns.drop(['average_token_length'],axis=1,inplace=True)
    pred_columns.drop(['kw_avg_min'],axis=1,inplace=True)
    pred_columns.drop(['LDA_00'],axis=1,inplace=True)
    pred_columns.drop(['LDA_04'],axis=1,inplace=True)
    pred_columns.drop(['global_subjectivity'],axis=1,inplace=True)
    pred_columns.drop(['global_sentiment_polarity'],axis=1,inplace=True)
    pred_columns.drop(['global_rate_positive_words'],axis=1,inplace=True)
    pred_columns.drop(['rate_positive_words'],axis=1,inplace=True)
    pred_columns.drop(['avg_positive_polarity'],axis=1,inplace=True)
    pred_columns.drop(['max_positive_polarity'],axis=1,inplace=True)

    prediction_var = pred_columns.columns

    print(prediction_var)

    train, test = train_test_split(data, test_size = 0.3,random_state=8)

    print("\n \n The shape of the Train Data is    :      ",train.shape)
    print("\n \n The shape of the Test Data is    :      ",test.shape)


    train_X = train[prediction_var]
    train_y=train['url']


    test_X= test[prediction_var] 
    test_y =test['url']   
    

    model = tree.DecisionTreeClassifier()
    model.fit(train_X,train_y)

    pred_y=model.predict(test_X)

        
    print('\n The predicted output of the maximum number of SHARES of Articles are            :              \t ',pred_y,'\n \n')



something_to_profile()


