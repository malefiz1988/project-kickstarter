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

def createModel(data=pd.DataFrame(), X=None, y=None, scoring='precision'):

    #Daten für Modelling
    if X == None and y == None:
        X, y = featureEngineering.preprocessData(data,test_size = 0.2, split = False)
    
    X,_,y,_ = train_test_split(X, y, train_size=0.1,stratify=y, random_state=RSEED)

    print("features engineered")

    svc_model = SVC()

    param_svc = {
                "C":[1.0],
                "kernel":['rbf'],
                "degree":[3],
                "gamma":['scale',0.1],
                "coef0":[0.0],
                "shrinking":[True],
                "probability":[False],
                "tol":[0.01],
                "cache_size":[200],
                "class_weight":[None],
                "verbose":[False],
                "max_iter":[-1],
                "decision_function_shape":['ovr'],
                "break_ties":[False],
                "random_state":[RSEED],
                }

    grid_svc = GridSearchCV(svc_model,
                               param_grid=param_svc,
                               cv=5, 
                               scoring=scoring,
                               verbose=5, 
                               n_jobs=-1)


    grid_svc.fit(X,y)
    print("model trained")

    svc_model = grid_svc.best_estimator_
    #y_log_pred_test = lg_model.predict(X_test)

    filename = './models/SVCModel.sav'
    pickle.dump(svc_model, open(filename, 'wb'))
    print("model saved")
    
    
    

if __name__ == "__main__":
    main()
    
def main():
    #Daten für EDA
    df = pd.read_csv('data/df_clean.csv')
    print("Daten geladen")
    
    createModel(data=df)