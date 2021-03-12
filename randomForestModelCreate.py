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

def createModel(data=pd.DataFrame(), X=None, y=None,scoring='precision',):

    #Daten für Modelling
    if X == None and y == None:
        X, y = featureEngineering.preprocessData(data,test_size = 0.2, split = False)
    
    X,_,y,_ = train_test_split(X, y, train_size=0.1,stratify=y, random_state=RSEED)

    print("features engineered")

    rf_model = RandomForestClassifier()

    param_rf = {
                "n_estimators":[100],
                "criterion":['gini'],
                "max_depth":[None],
                "min_samples_split":[10],
                "min_samples_leaf":[5],
                "min_weight_fraction_leaf":[0.0],
                "max_features":['auto'],
                "max_leaf_nodes":[None],
                "min_impurity_decrease":[0.0],
                "min_impurity_split":[None],
                "bootstrap":[True],
                "oob_score":[False],
                "n_jobs":[None],
                "random_state":[RSEED],
                "verbose":[0],
                "warm_start":[False],
                "class_weight":[None],
                "ccp_alpha":[0.0],
                "max_samples":[None],
                }

    grid_rf = GridSearchCV(rf_model,
                               param_grid=param_rf,
                               cv=5, 
                               scoring=scoring,
                               verbose=5, 
                               n_jobs=-1)

    grid_rf.fit(X,y)
    print("model trained")

    rf_model = grid_rf.best_estimator_
    #y_log_pred_test = lg_model.predict(X_test)

    filename = './models/RandomForestModel.sav'
    pickle.dump(rf_model, open(filename, 'wb'))
    print("model saved")
    
    
    

if __name__ == "__main__":
    main()
    
    
def main():
    #Daten für EDA
    df = pd.read_csv('data/df_clean.csv')
    print("Daten geladen")
    
    createModel(data=df)