# Replicability

## Data
We have feature data in all_features.csv, along with the codebook codebook.xlsx.

## Code
Models: run each model for all-feature data and citation-feature-reduced data, including using SMOTENC and minmax/standard scaler.
- LogisticRegression.py -> Logistic Regression
- MLP.py -> Multilayer Perceptron
- RandomForest.py -> Random Forest
- NB.py -> Naive Bayes
- SVM.py -> Support Vector Machine
- XGBoost.py -> XGBoost

XAI: use three XAI mehtods to address explainability for the model XGBoost
- SHAP PI XGBoost.py -> SHAP and Permutation feature importance
- SPLIME XGBoost.py -> SP-LIME

## Files
- feature-table.pdf: the single pdf page containing all features, the descriptions, the data types, and sources
- acc-fullreduced.pdf: the figure comparing the accuracy of 7 machine learning models on the full dataset and the reduced dataset (excluding citation-related features)
- pi-allfeatures.pdf: permutation importance (PI) values of XGBoost model with full features. Features with zero or negative PI values are not displayed.
- pi-nocitation.pdf: PI values of XGBoost model with reduced features (excluding citation-based features). 
