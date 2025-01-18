##Importing required libraries##

import numpy as np
import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split



import warnings
warnings.filterwarnings('ignore')




##loading the dataset##

ecom=pd.read_csv("online_shoppers_intention.csv")



ecom["Weekend"]=ecom["Weekend"].replace((True,False),(1,0))
ecom["Revenue"]=ecom["Revenue"].replace((True,False),(1,0))


con=ecom['VisitorType']=="Returning_Visitor"



ecom["Returning_Visitor"]=np.where(con,1,0)



ecom=ecom.drop(columns='VisitorType')


ordinal_encoder=OrdinalEncoder()


ecom['Month']=ordinal_encoder.fit_transform(ecom[['Month']])





result=ecom[ecom.columns[1:]].corr()['Revenue']

result1=result.sort_values(ascending=False)


X=ecom.drop(['Revenue'],axis=1)
y=ecom['Revenue']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


def model_pipeline(X, model): 
   
    n_c = X.select_dtypes(exclude=['object']).columns.tolist() 
    c_c = X.select_dtypes(include=['object']).columns.tolist() 
 
 
    numeric_pipeline = Pipeline([ 
            ('imputer', SimpleImputer(strategy='constant')), 
            ('scaler', MinMaxScaler()) 
    ]) 
 
 
    categorical_pipeline = Pipeline([ 
            ('encoder', OneHotEncoder(handle_unknown='ignore')) 
    ]) 
 
 
    preprocessor = ColumnTransformer([ 
            ('numeric', numeric_pipeline, n_c), 
            ('categorical', categorical_pipeline, c_c) 
    ], remainder='passthrough') 
 
 
 
    final_steps = [ 
            ('preprocessor', preprocessor), 
            ('smote', SMOTE(random_state=1)), 
            ('feature_selection', SelectKBest(score_func = chi2, k = 
    6)), 
            ('model', model) 
    ] 
 
    return IMBPipeline(steps = final_steps)


print("done")