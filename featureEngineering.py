# Import of relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn import preprocessing
import sklearn.metrics as sm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn import tree
import pickle

from sklearn.linear_model import LogisticRegression

RSEED = 42

def dropCols(df, drop_backers_count = True, drop_staff_pick = True):
    drop_cols = [
        'blurb',
        'creator',
        'currency',
        'currency_trailing_code',
        'fx_rate',
        'location',
        'name',
        'profile',
        'slug',
        'source_url',
        'static_usd_rate',
        'usd_type',
        "country",
        "is_starrable",
        'converted_pledged_amount',
        'pledged',
        'category_slug',
        'disable_communication',
        'usd_pledged',
        #"backers_count",
        #"staff_pick",
    ]
    if drop_backers_count == True: drop_cols.append("backers_count")
    if drop_staff_pick == True : drop_cols.append("staff_pick")
    
    
    df.drop(drop_cols, inplace=True, axis = 1)
    for c in df.columns: 
        if df[c].dtype == "object": 
            df[c] = df[c].astype("category") 
    df.dropna(inplace=True)
    return df

def toDay(time_delta):
    return round(time_delta/3600/24,2)

def dateTimeUpdate(df):
    campaign_length = df.deadline - df.created_at
    campaign_length = campaign_length.apply(toDay)
    campaign_length.name = "campaign_length"
    
    time_until_launch = df.launched_at - df.created_at
    time_until_launch = time_until_launch.apply(toDay)
    time_until_launch.name = "time_until_launch"
    
    time_launch2state = df.state_changed_at - df.launched_at
    time_launch2state = time_launch2state.apply(toDay)
    time_launch2state.name = "time_launch2state"
    
    time_state2deadline = df.deadline - df.state_changed_at
    time_state2deadline = time_state2deadline.apply(toDay)
    time_state2deadline.name = "time_state2deadline"
    
    time_launch2deadline = df.deadline - df.launched_at
    time_launch2deadline = time_launch2deadline.apply(toDay)
    time_launch2deadline.name = "time_launch2deadline"
    
    df = pd.concat([df, 
                    campaign_length,
                    time_until_launch, 
                    #time_launch2state,
                    #time_state2deadline,
                    #time_launch2deadline
                   ],
                  axis = 1
                  )
    df.drop([
        "created_at",
        "deadline",
        "launched_at",
        "state_changed_at",
            ],
        axis =1,
        inplace=True
    )
    return df






def prepDataFrameForPreprocessor(df, drop_backers_count = True, drop_staff_pick = True):
    """
        drops Columns
        creates time deltas and drops timestamps
        drops rows without state failed and successful
        transforms state column with labelencoder to 1 = successful and 0 = failed
        
        returns the prepared dataFrame
    """
    #df = pd.read_csv('data/df_clean.csv')
    df = dropCols(df,drop_backers_count,drop_staff_pick) #drop unnecessary columns
    df = dateTimeUpdate(df) #create time delta columns, add these and drop the original timesptamps
    #df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    df = df[(df['state']=='failed') | (df['state']=='successful')]
     
    #labelencode y from string to int
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(["failed","successful"])
        
    df["state"] = label_encoder.transform(df.state)
    
    return df
        
    
def fitPreprocessor(df):
    """
        Argument: DataFrame
    
        fits the preprocessor of the dataFrame
        imputer_num -> median
        std_scaler
        
        imputer_cat -> constant fill with missing
        1hot: ignore
        
        fits the preprocessor
                
        returns the preprocessor
        
        
    """

    for c in df.columns: 
        if df[c].dtype == "object": 
            df[c] = df[c].astype("category") 

    #get categorical features
    cat_features = list(df.columns[df.dtypes=='category'])
    #cat_features.remove('state')
    
    #get numerical features
    num_features = list(df.columns[df.dtypes!='category'])
    num_features.remove('state')
    
    # Pipeline for numerical features 
    num_pipeline = Pipeline([
        ('imputer_num', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])
    
    # Pipeline for categorical features
    cat_pipeline = Pipeline([
        ('imputer_cat', SimpleImputer(strategy='constant', fill_value='missing')),
        ('1hot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Complete pipeline for numerical and categorical features
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    
    #preprocess validationData
    X = df.drop("state", axis=1)
    
    preprocessor.fit(X)
    
    return preprocessor
    
    
    
    
    
    
    
    
#drop all rows not containing the prediction classes


# 
# #preprocessor = fitPreprocesser(test_size = test_size, split = split, drop_backers_count = drop_backers_count, drop_staff_pick = drop_staff_pick )
# X = preprocessor.transform(X)
# #X_test = preprocessor.transform(X_test)
# 
# #extract feature names
# #cat_cols= preprocessor.transformers_[1][1].named_steps["1hot"].get_feature_names(cat_features)
# #features = list(num_features) + list(cat_cols)
# 
# #split the data
# if (split == True):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,stratify=y, random_state=RSEED)
#     return X_train, X_test, y_train, y_test
# else:
#     return X, y
#