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
