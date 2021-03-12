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

def createModel(data=pd.DataFrame(), X=None, y=None,scoring = "precision"):

    #Daten für Modelling
    if X == None and y == None:
        X, y = featureEngineering.preprocessData(data,test_size = 0.2, split = False)
    
    X,_,y,_ = train_test_split(X, y, train_size=0.1,stratify=y, random_state=RSEED)

    print("features engineered")

    lg_model = LogisticRegression()

    param_lg = {"penalty":['l2'],
                "dual":[False],
                "tol":[0.01,1,5],
                "C":[1.0,0.5],
                "fit_intercept":[True],
                "intercept_scaling":[1],
                "class_weight":[None],
                "random_state":[RSEED],
                "solver":['lbfgs'],
                "max_iter":[1000],
                "multi_class":['auto'],
                "verbose":[0],
                "warm_start":[False],
                "n_jobs":[None],
                "l1_ratio":[None],
                }

    grid_lg = GridSearchCV(lg_model,
                               param_grid=param_lg,
                               cv=5, 
                               scoring=scoring,
                               verbose=5, 
                               n_jobs=-1)

    grid_lg.fit(X,y)
    print("model trained")

    lg_model = grid_lg.best_estimator_
    #y_log_pred_test = lg_model.predict(X_test)

    filename = './models/LogRegModel.sav'
    pickle.dump(lg_model, open(filename, 'wb'))
    print("model saved")
    
    
    

if __name__ == "__main__":
    main()
    
    
def main():
    #Daten für EDA
    df = pd.read_csv('data/df_clean.csv')
    print("Daten geladen")
    
    createModel(data=df)