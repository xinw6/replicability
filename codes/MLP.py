# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 23:56:06 2025

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

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTENC
#from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

data_type= 1
#data_type= 1  # automatic citation
#data_type= 2  # automatic no citation
#data_type= 3  # manual citation
#data_type= 4  # manual no citation

tuning = 0    # 1 yes 0 no

#**********************************************

filename = r"\all_features.csv"
filename_manual = r"/all_features_modify_add_manual.csv"
df = pd.read_csv(filename)
df_manual = pd.read_csv(filename_manual)
df_manual = df_manual.fillna(-1)


df = df[(df.citationVelocity != -1) & (df.influentialCitationCount != -1) & (df.Venue_subject_code  != -1) & (df.influentialReferencesCount != -1) &
           (df.reference_background != -1) & (df.reference_result != -1) & (df.reference_methodology != -1) & (df.subjectivity != -1) & (df.sentiment != -1)]
 
#df_manual = df_manual[(df_manual.citationVelocity != -1) & (df_manual.influentialCitationCount != -1) & (df_manual.Venue_subject_code  != -1) & (df_manual.influentialReferencesCount != -1) &
#          (df_manual.reference_background != -1) & (df_manual.reference_result != -1) & (df_manual.reference_methodology != -1) & (df_manual.subjectivity != -1) & (df_manual.sentiment != -1)]


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

if data_type==1 or 2:
    #model = MLPClassifier(alpha=0.001, max_iter=1000, activation='tanh', solver= 'adam')    # max_iter=1000 is important, 100 not good
    model = MLPClassifier(max_iter=1000)
    
    
elif data_type==3 or 4:
    model = MLPClassifier()    
    


model.fit(X_train, y_train)

# Predict the target variable for the resampled testing data
y_test_pred = model.predict(X_test)

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


#**************************************tuning parameters

if tuning ==1:
    param_grid = {
        'hidden_layer_sizes':  [(50,), (100,), (50, 30), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [100, 500, 1000]  
    }
    
    
    
    model = MLPClassifier(early_stopping=True)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1  
    )
    
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
