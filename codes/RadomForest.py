# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:47:55 2025

@author: weixi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
from pylab import rcParams
import seaborn as sb
from scipy.stats.stats import kendalltau
import io
#from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTENC
#from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report 
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from xgboost import plot_importance
# smotnc logregression

data_type= 1
#data_type= 1  # automatic citation
#data_type= 2  # automatic no citation
#data_type= 3  # manual citation
#data_type= 4  # manual no citation

tuning = 0    # 1 yes 0 no

smotenc_use = 0    # 1 yes 0 no   # when use smotenc, no tuning

scaler_use = 0     # 1 yes 0 no

#Scaler= MinMaxScaler
Scaler= StandardScaler
#**********************************************

filename = r"\all_features.csv"
filename_manual = r"/all_features_modify_add_manual.csv"
df = pd.read_csv(filename)
df_manual = pd.read_csv(filename_manual)
df_manual = df_manual.fillna(-1)


df = df[(df.citationVelocity != -1) & (df.influentialCitationCount != -1) & (df.Venue_subject_code  != -1) & (df.influentialReferencesCount != -1) &
           (df.reference_background != -1) & (df.reference_result != -1) & (df.reference_methodology != -1) & (df.subjectivity != -1) & (df.sentiment != -1)]
 
df_manual = df_manual[(df_manual.citationVelocity != -1) & (df_manual.influentialCitationCount != -1) & (df_manual.Venue_subject_code  != -1) & (df_manual.influentialReferencesCount != -1) &
          (df_manual.reference_background != -1) & (df_manual.reference_result != -1) & (df_manual.reference_methodology != -1) & (df_manual.subjectivity != -1) & (df_manual.sentiment != -1)]

df = df.drop(['coCite3', 'num_significant', 'normalized_citations', 'avg_high_inf_cites', 'citations_background'], axis = 1) 

#********* pad real_p, p_val_range with mean values
for index, row in df.iterrows():
    if row['real_p']< -2:
        df['real_p'][index]=1
    if row['p_val_range']> 1:
        df['p_val_range'][index]=0    
        
        

mean_real_p = df[df['real_p'] != 1]['real_p'].mean()
print(f"Mean for real_p (excluding 1): {mean_real_p}")

mean_p_val_range = df[df['p_val_range'] != 0]['p_val_range'].mean()
print(f"Mean for p_val_range (excluding 1): {mean_p_val_range}")


for index, row in df.iterrows():
    if row['real_p']==1:
        df['real_p'][index]=mean_real_p
        df['p_val_range'][index]=mean_p_val_range
        
        
#**********        
"""
for index, row in df_manual.iterrows():
    if row['real_p']< -2:
        df_manual['real_p'][index]=1
    if row['p_val_range']> 1:
        df_manual['p_val_range'][index]=0          
        
        

mean_real_p_m = df_manual[df_manual['real_p'] != 1]['real_p'].mean()
print(f"Mean (excluding 1): {mean_real_p_m}")

mean_p_val_range_m = df_manual[df_manual['p_val_range'] != 0]['p_val_range'].mean()
print(f"Mean for p_val_range (excluding 1): {mean_p_val_range_m}")

for index, row in df_manual.iterrows():
    if row['real_p']==1:
        df_manual['real_p'][index]=mean_real_p_m        
        df_manual['p_val_range'][index]=mean_p_val_range_m

"""
#*********************data
if data_type== 1:        # automatically extracted, citation data included

   
    X = df.drop(['y', 'doi', 'title', 'real_p_sign', 'Venue_subject'], axis = 1)
    
    y = df['y']
    
    print("data type 1")
elif data_type== 2:  

    #df = df.drop(['Venue_CiteScore'], axis = 1)
    X = df.drop(['y', 'doi', 'title', 'real_p_sign', 'Venue_subject'], axis = 1)
    #X = X.drop(['num_citations', 'self_citations', 'citationVelocity', 'influentialCitationCount', 'normalized_citations', 'citations_background', 'citations_result', 'citations_methodology',
    #         'citations_next', 'coCite2', 'coCite3', 'avg_auth_cites', 'avg_high_inf_cites', 'Venue_Citation_Count', 'Venue_Percent_Cited', 'Venue_CiteScore'], axis = 1)
    X = X.drop(['num_citations', 'self_citations', 'citationVelocity', 'influentialCitationCount', 'normalized_citations', 'citations_background', 'citations_result', 'citations_methodology',
             'citations_next', 'coCite2', 'coCite3' ], axis = 1)
    
    y = df['y']
    print("data type 2")
    
elif data_type== 3:
    
    X_manual = df_manual.drop(['y', 'doi', 'title', 'real_p_sign', 'Venue_subject', 'pdf_filename'], axis = 1)
    y = df_manual['y']
    X =X_manual
    print("data type 3")
    
elif data_type== 4:

    X_manual = df_manual.drop(['y', 'doi', 'title', 'real_p_sign', 'Venue_subject', 'pdf_filename'], axis = 1)
   # X_manual = X_manual.drop(['num_citations', 'self_citations', 'citationVelocity', 'influentialCitationCount', 'normalized_citations', 'citations_background', 'citations_result', 'citations_methodology',
    #         'citations_next', 'coCite2', 'coCite3', 'avg_auth_cites', 'avg_high_inf_cites', 'Venue_Citation_Count', 'Venue_Percent_Cited', 'Venue_CiteScore'], axis = 1)
    X_manual = X_manual.drop(['num_citations', 'self_citations', 'citationVelocity', 'influentialCitationCount', 'normalized_citations', 'citations_background', 'citations_result', 'citations_methodology',
             'citations_next', 'coCite2', 'coCite3' ], axis = 1)

    y = df_manual['y']
    X =X_manual
    print("data type 4")
    

    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#*******************************************model

if data_type==1:
    model = RandomForestClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=5, max_features= 'sqrt',
                           n_estimators=200, random_state=42)


elif data_type== 2:
    model = RandomForestClassifier(max_depth=10, min_samples_leaf= 1, min_samples_split=10,  max_features= 'sqrt',
                           n_estimators=50, random_state=42)

elif data_type==3:
    #model = RandomForestClassifier(max_depth=10, max_features='log2', min_samples_leaf=2,
                 #          n_estimators=50, random_state=42)
    
    model = RandomForestClassifier(min_samples_leaf=2, n_estimators=50, random_state=42)
    
    
elif data_type== 4:
    #model = RandomForestClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=5, max_features= 'sqrt',
                     #      n_estimators=50, random_state=42)
    model = RandomForestClassifier(random_state=42)




if smotenc_use == 0:
    model.fit(X_train, y_train)

elif smotenc_use == 1:
    
    categorical_cols = ["openaccessflag", "sentiment", "funded", "Venue_subject_code"]
    feature_names = [X.columns.get_loc(col) for col in categorical_cols]
    #feature_names=[X.columns.get_loc("openaccessflag"), X.columns.get_loc("sentiment"), X.columns.get_loc("funded"), X.columns.get_loc("Venue_subject_code")]
    print(feature_names)
    smotenc = SMOTENC(categorical_features=feature_names, random_state=42)
    # Apply SMOTE to oversample the minority class
    
    X_train_resampled, y_train_resampled = smotenc.fit_resample(X_train, y_train)
    
    # Print the class distribution before and after oversampling
    print("Before oversampling:")
    print(y_train.value_counts())
    print("\nAfter oversampling:")
    print(y_train_resampled.value_counts())
    
    if scaler_use == 0:
        model.fit(X_train_resampled, y_train_resampled)


    elif scaler_use == 1:
        
        ### SMOTENC with Scaler
        if Scaler== MinMaxScaler:
            # Create a MinMaxScaler object
            scaler = MinMaxScaler()
        elif Scaler== StandardScaler:
            scaler = StandardScaler()
            
        # Scale the training features    
        X_train_resampled_con = X_train_resampled.drop(["openaccessflag", "sentiment", "funded", "Venue_subject_code"], axis = 1)
        con_names = X_train_resampled_con.columns
        X_train_resampled_cat = X_train_resampled[["openaccessflag", "sentiment", "funded", "Venue_subject_code"]]
        
     
        #X_train_scaled = scaler.transform(X_train)
        X_train_resampled_con_scaled = scaler.fit_transform(X_train_resampled_con)
        X_train_resampled_con_scaled = pd.DataFrame(X_train_resampled_con_scaled)
        X_train_resampled_con_scaled.columns = con_names
        X_train_resampled_scaled =X_train_resampled_con_scaled.join(X_train_resampled_cat)
        X_train_resampled_scaled = X_train_resampled_scaled.values

        model.fit(X_train_resampled_scaled, y_train_resampled)
        
        #X_test = scaler.transform(X_test)   
        
        X_test_con = X_test.drop(["openaccessflag", "sentiment", "funded", "Venue_subject_code"], axis = 1)
        X_test_con = X_test_con.reset_index(drop=True)
        con_names = X_test_con.columns
        X_test_cat = X_test[["openaccessflag", "sentiment", "funded", "Venue_subject_code"]]
        X_test_cat = X_test_cat.reset_index(drop=True)
        
        X_test_con_scaled = scaler.transform(X_test_con)
        X_test_con_scaled = pd.DataFrame(X_test_con_scaled)
        X_test_con_scaled.columns = con_names
        X_test_scaled =X_test_con_scaled.join(X_test_cat)   # contain NaN after join 因为index不一样，需要reset index

    
    
    
# Predict the target variable for the resampled testing data
if smotenc_use == 0:
    y_test_pred = model.predict(X_test)
elif smotenc_use == 1 and scaler_use == 0:
    y_test_pred = model.predict(X_test)  
    print('smotenc!!')# smotenc
elif smotenc_use == 1 and scaler_use == 1:
    y_test_pred = model.predict(X_test_scaled) 
    print('sclaer!!')# scaler


print('\n')
print(pd.Series(y_test_pred).value_counts())     

# Calculate the accuracy of the model on the resampled testing data
accuracy_test = accuracy_score(y_test, y_test_pred)
print("\nModel Accuracy on original Testing Data:", accuracy_test)

# Calculate the precision, recall, and F1 score on the resampled testing data
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
print("Model Precision on original Testing Data:", precision_test)
print("Model Recall on original Testing Data:", recall_test)
print("Model F1 Score on original Testing Data:", f1_test)


#************************************** plot feature importance

#plot_importance(model, max_num_features=10)
#plt.show()


#**************************************tuning parameters

if tuning ==1:
    
    
    
    
    
    param_grid = { 'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'max_features': ['sqrt', 'log2'], 
                  'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4] }
    
    
    model = RandomForestClassifier(random_state=42) 
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train) 
    
    
    # show best parameters
    param = grid_search.best_params_
    print('\n')
    print(param)

    # best model
    best_model = grid_search.best_estimator_ 
    print('best model:', best_model)
    # Predict for the testing data
    y_pred = best_model.predict(X_test) 
    
    # F1、Precision、Recall on testing set
    print(classification_report(y_test, y_pred))
    
    # F1、Precision、Recall on testing set
    print(f"Test F1: {f1_score(y_test, y_pred):.3f}") 
    print(f"Test Precision: {precision_score(y_test, y_pred):.3f}") 
    print(f"Test Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")