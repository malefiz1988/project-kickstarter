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

def loadData(RS=42):
    i=0
    df_raw = pd.DataFrame()
    for f in os.listdir("./data"):
        if f[:4] == "Kick":
            da = pd.read_csv("./data/" + str(f))
            df_raw = pd.concat([df_raw, da], axis = 0)
            i+=1
            print(i,f)
    df_raw = df_raw.drop_duplicates(subset='id', keep='first')
    df_raw.reset_index(drop=True, inplace=True)
    df, df_backUp = train_test_split(df_raw,test_size=0.1,stratify=df_raw.state,random_state=RS)
    #df_backUp.to_csv("./data/df_backUp.csv")
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
        'urls',
        "spotlight"
    ]
    df.drop(drop_cols, inplace=True, axis = 1)
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

def categorizeObjects(df):
    for c in df.columns: 
        if df[c].dtype == "object": 
            df[c] = df[c].astype("category") 
    df.dropna(inplace=True)
    return df

def saveReadCleanData(RSEED=42):
    """
    Opens the filepath ./data/df_clean.csv, reads in all the files starting with 'Kick', drops Trash columns, adds a column for grouped Countries (most important) and turns the category column into two seperate columns having read the category_name and slug_name from the original column. category is then dropped. 
    
    Nothing is returned but two datasets are saved in following location ./data/df_clean.csv and ./data/df_backUp.csv
    
    df is the dataset to work with, while df_backUp represents the testingData for the end testing.
    """
    
    df, df_backUp = loadData(RSEED)

    df = catCleaner(df)
    df = groupCountries(df)
    df = dropTrashCols(df)
    df = categorizeObjects(df)

    df_backUp = catCleaner(df_backUp)
    df_backUp = groupCountries(df_backUp)
    df_backUp = dropTrashCols(df_backUp)
    df_backUp = categorizeObjects(df_backUp)

    df.to_csv("./data/df_clean.csv",index=False)
    df_backUp.to_csv("./data/df_backUp.csv",index=False)

def returnReadCleanData(RSEED=42):
    """
    Opens the filepath ./data/df_clean.csv, reads in all the files starting with 'Kick', drops Trash columns, adds a column for grouped Countries (most important) and turns the category column into two seperate columns having read the category_name and slug_name from the original column. category is then dropped. 
    
    Two dataFrames are returned -> return df, df_backUp
    
    df is the dataset to work with, while df_backUp represents the testingData for the end testing.
    """
    df, df_backUp = loadData(RSEED)

    df = catCleaner(df)
    df = groupCountries(df)
    df = dropTrashCols(df)
    df = categorizeObjects(df)

    df_backUp = catCleaner(df_backUp)
    df_backUp = groupCountries(df_backUp)
    df_backUp = dropTrashCols(df_backUp)
    df_backUp = categorizeObjects(df_backUp)

    return df, df_backUp


    
    
def main():
    print("los")
    RSEED=42
    saveReadCleanData(RSEED)
    
    
if __name__ == "__main__":
    main()