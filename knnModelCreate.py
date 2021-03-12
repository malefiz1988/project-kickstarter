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

    knn_model = KNeighborsClassifier()

    #Grid search Parameters
    param_knn = {
                "n_neighbors":[5,20,50],
                "weights":['uniform'],
                "algorithm":['auto'],
                "leaf_size":[30],
                "p":[2,1],
                "metric":['minkowski'],
                "metric_params":[None],
                "n_jobs":[None],
                   }

    grid_knn = GridSearchCV(knn_model,
                               param_grid=param_knn,
                               cv=5, 
                               scoring=scoring,
                               verbose=1, 
                               n_jobs=-1)

    grid_knn.fit(X,y)
    print("model trained")

    knn_model = grid_knn.best_estimator_
    #y_log_pred_test = lg_model.predict(X_test)

    filename = './models/KNNModel.sav'
    pickle.dump(knn_model, open(filename, 'wb'))
    print("model saved")
    
    
    

if __name__ == "__main__":
    main()
    
def main():
    #Daten für EDA
    df = pd.read_csv('data/df_clean.csv')
    print("Daten geladen")
    
    createModel(data=df)