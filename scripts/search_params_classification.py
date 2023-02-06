import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import f1_score

def f1_model_score(estimator, X, y):
    y_hat = estimator.predict(X)
    return f1_score(y, y_hat)


random_state = 109 # Randomly generated number 1-1000

xgb_base_model = XGBClassifier(n_estimators=10, max_depth=3, n_jobs=-1, random_state=random_state)
xgb_search_params = {'estimator': XGBClassifier(n_jobs=-1, random_state=random_state),
                     'search_spaces': {'n_estimators': Integer(10, 100, 'uniform'),
                                       'max_depth': np.arange(1, 12, 2),
                                       'min_child_weight': Real(0.01, 20, 'log-uniform'),
                                       'gamma': Real(0.01, 100, 'log-uniform'),
                                       'reg_alpha': Real(0.01, 10, 'log-uniform'),
                                       'reg_lambda': Real(0.01, 10, 'log-uniform')},
                     'n_iter': 100,
                     'scoring': 'f1',
                     'refit': True,
                     'random_state': random_state}

logreg_base_model = LogisticRegression(penalty='elasticnet', solver='saga', n_jobs=-1, random_state=random_state)
logreg_search_params = {'estimator': LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000, n_jobs=-1, random_state=random_state),
                        'search_spaces': {'C': Real(0.001, 100, 'log-uniform'),
                                          'l1_ratio': Real(0, 1, 'uniform')},
                        'n_iter': 100,
                        'scoring': 'f1',
                        'refit': True,
                        'random_state': random_state}
