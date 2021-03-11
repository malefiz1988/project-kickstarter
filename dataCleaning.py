from xgboost import XGBClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
import seaborn as sns
import os

import pandas as pd
from pandas_profiling import ProfileReport
RSEED=42

from IPython.display import display
pd.options.display.max_columns = None

def loadData():
    i=0
    df_raw = pd.DataFrame()
    for f in os.listdir("./data"):
        if f[:4] == "Kick":
            df = pd.read_csv("./data/" + str(f))
            df_raw = pd.concat([df_raw, df], axis = 0)
            i+=1
            print(i,f)
    return df_raw

def cat2name(cat):
    replaceString = '}{"'
    for c in replaceString: cat = cat.replace(c,"")
    dic = {}
    d = cat.split(",")
    for i in d: dic[i.split(":")[0]] = i.split(":")[1]
    return dic["name"]

def cat2slug(cat):
    replaceString = '}{"'
    for c in replaceString: cat = cat.replace(c,"")
    dic = {}
    d = cat.split(",")
    for i in d: dic[i.split(":")[0]] = i.split(":")[1]
    return dic["slug"]

def catCleaner(df):
    name = df.category.apply(cat2name)
    name.name = "category_name"
    slug = df.category.apply(cat2slug)
    slug.name = "category_slug"
    df.drop("category", inplace=True, axis =1)
    df = pd.concat([df, name, slug], axis= 1)
    return df

def prepForModel(df):
    drop_cols = ["blurb","creator","currency","currency_symbol","currency_trailing_code","current_currency","friends","fx_rate","id","is_backing","is_starred",
             "location","name","permissions","photo","profile","slug","source_url","static_usd_rate","urls","usd_type"]
    df.drop(drop_cols, inplace=True, axis = 1)
    for c in df.columns: 
        if df[c].dtype == "object": 
            df[c] = df[c].astype("category") 
    df.dropna(inplace=True)
    return df

#clf = LogisticRegression(random_state=RSEED,tol = 0.01, max_iter=50)

def trainEval(df, clf):
    y = df_raw.state
    X = df_raw.drop("state", axis=1)

    X = pd.get_dummies(X)

    X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=RSEED)

    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(sm.classification_report(y_train, y_pred_train));

    print(sm.classification_report(y_test, y_pred_test))

da = loadData()
da = catCleaner(da)
da = prepForModel(da)
clf = LogisticRegression(random_state=RSEED,tol = 0.01, max_iter=50)
trainEval(da, clf)


