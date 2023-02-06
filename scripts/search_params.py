import numpy as np
from models import RansacElasticNet
from sklearn.linear_model import RANSACRegressor, ElasticNet
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import r2_score

def r2_model_score(estimator, X, y):
    y_hat = estimator.predict(X)
    return r2_score(y, y_hat)


#random_state = 331
random_state = 874


xgb_base_model = XGBRegressor(n_estimators=50, max_depth=3, n_jobs=-1, random_state=random_state)
#xgb_search_params = {'estimator': XGBRegressor(n_jobs=-1, random_state=random_state),
#                     'search_spaces': {'n_estimators': Integer(10, 200, 'log-uniform'),
#                                       'max_depth': np.arange(1, 12, 2),
#                                       'learning_rate': Real(0.001, 1, 'log-uniform'),
#                                       'colsample_bytree': Real(0.1, 1, 'uniform'),
#                                       'colsample_bylevel': Real(0.1, 1, 'uniform'),
#                                       'colsample_bynode': Real(0.1, 1, 'uniform'),
#                                       'min_child_weight': Real(0.1, 20, 'uniform'),
#                                       'gamma': Real(0, 20, 'uniform'),
#                                       'subsample': Real(0.5, 1, 'uniform'),
#                                       'reg_alpha': Real(0.01, 10, 'log-uniform'),
#                                       'reg_lambda': Real(0.01, 10, 'log-uniform')},
#                     'n_iter': 100,
#                     'scoring': r2_model_score,
#                     'refit': True,
#                     'random_state': random_state}
xgb_search_params = {'estimator': XGBRegressor(base_score=1., n_jobs=-1, random_state=random_state),
                     'search_spaces': {'n_estimators': Integer(10, 100, 'uniform'),
                                       'max_depth': np.arange(1, 12, 2),
                                       'min_child_weight': Real(0.01, 20, 'log-uniform'),
                                       'gamma': Real(0.01, 100, 'log-uniform'),
                                       'reg_alpha': Real(0.01, 10, 'log-uniform'),
                                       'reg_lambda': Real(0.01, 10, 'log-uniform')},
                     'n_iter': 100,
                     'scoring': r2_model_score,
                     'refit': True,
                     'random_state': random_state}


elasticnet_base_model = ElasticNet(random_state=random_state, max_iter=10000)
elasticnet_search_params = {'estimator': ElasticNet(random_state=random_state, max_iter=10000),
                            'search_spaces': {'alpha': Real(0.001, 100, 'log-uniform'),
                                              'l1_ratio': Real(0, 1, 'uniform')},
                            'n_iter': 100,
                            'scoring': r2_model_score,
                            'refit': True,
                            'random_state': random_state}

#elasticnet_search_params = {'estimator': RansacElasticNet(max_iter=10000, random_state=random_state),
#                            'search_spaces': {'alpha': Real(0.001, 10, 'log-uniform'),
#                                              'l1_ratio': np.arange(0, 1.1, 0.1)},
#                            'n_iter': 50,
#                            'scoring': r2_model_score,
#                            'refit': True,
#                            'random_state': random_state}
