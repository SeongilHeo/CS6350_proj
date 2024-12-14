from utils import *
import numpy as np

X_train, y_train = load_train()
X_test, y_test = load_ftest()

# X_train.replace('?',np.nan,inplace=True)
# X_test.replace('?',np.nan,inplace=True)

# X_train.dropna()
# X_test.dropna()

# X_train.loc[X_train['native.country']!='United-States', 'native.country'] = 'Non-US'
# X_train.loc[X_train['native.country'] == 'United-States', 'native.country'] = 'US'
# X_test.loc[X_test['native.country']!='United-States', 'native.country'] = 'Non-US'
# X_test.loc[X_test['native.country'] == 'United-States', 'native.country'] = 'US'


X_train['marital.status'] = X_train['marital.status'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
X_train['marital.status'] = X_train['marital.status'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')

X_test['marital.status'] = X_test['marital.status'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
X_test['marital.status'] = X_test['marital.status'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')

X_train, X_test = ecode_onehot(X_train,X_test)

import lightgbm as lgb

model = lgb.LGBMClassifier(class_weight='balanced', max_depth=11, n_estimators=132)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)[:, 1] 
accuracy_score(y_test,y_pred)

export_pred(y_pred)