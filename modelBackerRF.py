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

def createModel(data, scoring='precision', drop_backers_count = False, drop_staff_pick = True):
    
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
    rf_model = RandomForestClassifier()

    param_rf = {
                "n_estimators":[1000],
                "criterion":['entropy'],
                "max_depth":[None],
                "min_samples_split":[2],
                "min_samples_leaf":[1],
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

    filename = './models/modelBackerRF.sav'
    pickle.dump(rf_model, open(filename, 'wb'))
    print("model saved")
    
    filename = './models/preprocessorBackerRF.sav'
    pickle.dump(preprocessor, open(filename, 'wb'))
    print("preprocessor saved")
    
    
def main():
    #Daten für EDA
    df = pd.read_csv('data/df_clean.csv')
    print("Daten geladen")
    
    createModel(data=df)

if __name__ == "__main__":
    main()
    
    
