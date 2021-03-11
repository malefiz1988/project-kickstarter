from scipy.sparse.construct import rand
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
from IPython.display import display
from datetime import datetime
#pd.options.display.max_columns = None

RSEED=42

def loadData(RSEED=42):
    i=0
    df_raw = pd.DataFrame()
    for f in os.listdir("./data"):
        if f[:4] == "Kick":
            da = pd.read_csv("./data/" + str(f))
            df_raw = pd.concat([df_raw, da], axis = 0)
            i+=1
            print(i,f)
    df_raw.reset_index(drop=True, inplace=True)
    df, df_backUp = train_test_split(df_raw,test_size=0.1,stratify=df_raw.state,random_state=RSEED)
    df_backUp.to_csv("./data/df_backUp.csv")
    return df, df_backUp

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

def timeline(t, m="dt"):
    if m == "days":
        return int(t/24/60/60)
    elif m == "months":
        return int(t/24/60/60/30.4167)
    elif m == "year":
        return int(t/24/60/60/30.4167/12) + 1970
    elif m == "dt":
        return datetime.fromtimestamp(t)

def prepForModel(df):
    drop_cols = [
        'blurb',
        'creator',
        'currency',
        'currency_symbol',
        'currency_trailing_code',
        'current_currency',
        'friends',
        'fx_rate',
        'id',
        'is_backing',
        'is_starred',
        'location',
        'name',
        'permissions',
        'photo',
        'profile',
        'slug',
        'source_url',
        'static_usd_rate',
        'urls',
        'usd_type'
    ]
    df.drop(drop_cols, inplace=True, axis = 1)
    for c in df.columns: 
        if df[c].dtype == "object": 
            df[c] = df[c].astype("category") 
    df.dropna(inplace=True)
    return df

def groupCountries(df):
    map_dictionary ={
        "DE" : "Europe",
        "FR" :"Europe",
        "IT" : "Europe",
        "ES":"Europe",
        "NL":"Europe",
        "SE": "Europe",
        "DK": "Europe",
        "BE": "Europe",
        "NO": "Europe",
        "AT": "Europe",
        "LU": "Europe", 
        "CH": "Europe", 
        "IE": "Europe", 
        "JP": "Asia", 
        "HK": "Asia",
        "SG": "Asia", 
        "MX": "Other",
        "NZ": "Other",
        "AU": "Other", 
        "US": "US",
        "GB": "GB",
        "CA": "CA"
    } 
    df['cgrouped']  = df['country'].map(map_dictionary)
    return df

def dropTrashCols(df):
    drop_cols = [
        'is_backing',
        'is_starred',
        'currency_symbol',
        'current_currency', 
        'friends',
        'id',
        'permissions',
        'photo',
        'urls'
    ]
    df.drop(drop_cols, inplace=True, axis = 1)
    return df
    
def categorizeObjects(df):
    for c in df.columns: 
        if df[c].dtype == "object": 
            df[c] = df[c].astype("category") 
    df.dropna(inplace=True)
    return df

def trainEval(df, clf):
    y = df.state
    X = df.drop("state", axis=1)

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


