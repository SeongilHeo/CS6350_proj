# 라이브러리 임포트
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split

X, y = load_train()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.replace('?','Unknown',inplace=True)
X_test.replace('?','Unknown',inplace=True)

X_train['workclass'].replace(['Federal-gov','Local-gov','State-gov'],'Gov',inplace=True)
X_train['workclass'].replace(['Self-emp-not-inc','Self-emp-inc'],'Self', inplace=True)
X_train['workclass'].replace(['Never-worked','Without-pay', 'Unknown'],'Other/Unknown ',inplace=True)

X_test['workclass'].replace(['Federal-gov','Local-gov','State-gov'],'Gov', inplace=True)
X_test['workclass'].replace(['Self-emp-not-inc','Self-emp-inc'],'Self', inplace=True)
X_test['workclass'].replace(['Never-worked','Without-pay', 'Unknown'],'Other/Unknown', inplace=True)
# -------------------------------------------------------------------------------------------------------------------
X_train['education'].replace(['Preschool', '1st-4th', '5th-6th', '7th-8th'], 'Elem', inplace=True)
X_train['education'].replace(['9th', '10th', '11th', '12th', 'HS-grad'], 'HS-grad', inplace=True)
X_train['education'].replace(['Assoc-acdm', 'Assoc-voc'], 'Assoc', inplace=True)
# X_train['education'].replace(['Bachelors', 'Masters', 'Doctorate', 'Prof-school'], 'Graduate', inplace=True)
# 'Some-college',
X_test['education'].replace(['Preschool', '1st-4th', '5th-6th', '7th-8th'], 'Elem', inplace=True)
X_test['education'].replace(['9th', '10th', '11th', '12th', 'HS-grad'], 'HS-grad', inplace=True)
X_test['education'].replace(['Assoc-acdm', 'Assoc-voc'], 'Assoc', inplace=True)
# X_test['education'].replace(['Bachelors', 'Masters', 'Doctorate', 'Prof-school'], 'Graduate', inplace=True)
# 'Some-college',
# -------------------------------------------------------------------------------------------------------------------
#  , 'Adm-clerical'
X_train['occupation'].replace(['Adm-clerical', 'Tech-support'], 'White-Collar', inplace=True)
X_train['occupation'].replace(['Craft-repair', 'Machine-op-inspct', 'Farming-fishing'],'Blue-Collar', inplace=True)
X_train['occupation'].replace(['Priv-house-serv', 'Handlers-cleaners','Other-service','Transport-moving',], 'Manual-Labor', inplace=True)
X_train['occupation'].replace(['Prof-specialty','Protective-serv','Exec-managerial'], 'Professional', inplace=True)
X_train['occupation'].replace(['Armed-Forces', 'Unknown'], 'Other/Unknown', inplace=True)

X_test['occupation'].replace(['Adm-clerical', 'Tech-support'], 'White-Collar', inplace=True)
X_test['occupation'].replace(['Craft-repair','Machine-op-inspct', 'Farming-fishing'],'Blue-Collar', inplace=True)
X_test['occupation'].replace(['Priv-house-serv', 'Handlers-cleaners','Other-service','Transport-moving',], 'Manual-Labor', inplace=True)
X_test['occupation'].replace(['Prof-specialty','Protective-serv','Exec-managerial'], 'Professional', inplace=True)
X_test['occupation'].replace(['Armed-Forces', 'Unknown'], 'Other/Unknown', inplace=True)

# # -------------------------------------------------------------------------------------------------------------------

X_train['marital.status'].replace(['Married-civ-spouse','Married-AF-spouse'], 'Married', inplace=True)
X_train['marital.status'].replace(['Divorced', 'Separated','Widowed', 'Married-spouse-absent'], 'Previously-Married', inplace=True)
# 'Never-married', 
X_test['marital.status'].replace(['Married-civ-spouse','Married-AF-spouse','Married-spouse-absent'], 'Married', inplace=True)
X_test['marital.status'].replace(['Divorced', 'Separated','Widowed', 'Married-spouse-absent'], 'Previously-Married', inplace=True)
# 'Never-married', 
# -------------------------------------------------------------------------------------------------------------------

X_train['relationship'].replace(['Not-in-family', 'Unmarried', 'Other-relative'],'Not-in-family', inplace=True)
# X_train['relationship'].replace(['Husband', 'Own-child', 'Wife'], 'Family', inplace=True)
X_test['relationship'].replace(['Not-in-family', 'Unmarried', 'Other-relative'],'Not-in-family', inplace=True)
# X_test['relationship'].replace(['Husband', 'Own-child', 'Wife'], 'Family', inplace=True)
# -------------------------------------------------------------------------------------------------------------------

X_train.loc[X_train['native.country']!=' United-States', 'native.country'] = 'Non-US'
X_train.loc[X_train['native.country'] == ' United-States', 'native.country'] = 'US'
X_test.loc[X_test['native.country']!=' United-States', 'native.country'] = 'Non-US'
X_test.loc[X_test['native.country'] == ' United-States', 'native.country'] = 'US'

X_train, X_test = ecode_onehot(X_train,X_test)

import lightgbm as lgb

model = lgb.LGBMClassifier(
    class_weight='balanced'
)

param_grid = {
    'n_estimators': list(range(1,200)),
    'max_depth': list(range(4,100)),
    'num_leaves': list(range(2,100)),
}

grid_search = GridSearchCV(
    estimator=model, 
    param_grid=param_grid, 
    cv=5,  
    n_jobs=-1, 
    scoring='accuracy', 
    verbose=5
)

grid_search.fit(X_train, y_train)

print(f"Optimal hyperparameters: {grid_search.best_params_}")
print(f"Optimal model acc: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\nTest acc: {accuracy_score(y_test, y_pred):.4f}")
print("\nReport:")
print(classification_report(y_test, y_pred))
