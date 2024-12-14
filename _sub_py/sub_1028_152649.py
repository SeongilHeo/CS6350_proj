from utils import *

X_train, y_train = load_train()
X_test = load_test()
X_test, y_test = load_ftest()

X_train, X_test = ecode_onehot(X_train,X_test)

import lightgbm as lgb

model = lgb.LGBMClassifier(class_weight='balanced', max_depth=11)
model.fit(X_train, y_train)

y_pred = model.predict_proba(X_test)[:, 1] 
accuracy_score(y_test,y_pred)

export_pred(y_pred)