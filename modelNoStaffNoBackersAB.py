# Import of relevant packages
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier

from sklearn.model_selection import train_test_split

import pickle

import featureEngineering

RSEED = 42

def createModel(data, scoring='precision', drop_backers_count = True, drop_staff_pick = True):
    
    data = featureEngineering.prepDataFrameForPreprocessor(data, drop_backers_count = drop_backers_count, drop_staff_pick = drop_staff_pick)
    
    print(data.columns)
    
    preprocessor = featureEngineering.fitPreprocessor(data)
    
    X = data.drop("state", axis=1)
    y = data["state"] 
    
    print("before preprocessing")
    
    X = preprocessor.transform(X)
    
    print("features engineered")
    print("X",X.shape)
    print("y",y.shape)

    ab_model = AdaBoostClassifier()

    param_ab = {            
                "base_estimator":[None],
                "n_estimators":[100],
                "learning_rate":[1.0],
                "algorithm":['SAMME.R'],
                "random_state":[RSEED],
                }

    grid_ab = GridSearchCV(ab_model,
                               param_grid=param_ab,
                               cv=5, 
                               scoring=scoring,
                               verbose=5, 
                               n_jobs=-1)

    grid_ab.fit(X,y)
    print("model trained")

    ab_model = grid_ab.best_estimator_
    #y_log_pred_test = lg_model.predict(X_test)

    filename = './models/modelNoStaffNoBackersAB.sav'
    pickle.dump(ab_model, open(filename, 'wb'))
    print("model saved")
    
    filename = './models/preprocessorNoStaffNoBackersAB.sav'
    pickle.dump(preprocessor, open(filename, 'wb'))
    print("preprocessor saved")
    

def main():
    #Daten für EDA
    df = pd.read_csv('data/df_clean.csv')
    print("Daten geladen")
    
    createModel(data=df)

    
    
if __name__ == "__main__":
    main()
    
