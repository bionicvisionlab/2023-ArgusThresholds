import numpy as np
from pandas import isnull
import sklearn.metrics as sklm


def calc_r2(y_true, y_pred):
    """Calculates the R2 score between y_true and y_pred"""
    idx = ~isnull(y_pred)
    if np.sum(~idx) > 0:
        print("Warning: %d NaN values found and excluded" % np.sum(~idx))
    return sklm.r2_score(y_true[idx], y_pred[idx])


def calc_rmse(y_true, y_pred):
    """Calculates the root mean squared error"""
    idx = ~isnull(y_pred)
    if np.sum(~idx) > 0:
        print("Warning: %d NaN values found and excluded" % np.sum(~idx))
    return np.sqrt(sklm.mean_squared_error(y_true[idx], y_pred[idx]))


def calc_mae(y_true, y_pred):
    """Calculates the mean absolute error"""
    idx = ~isnull(y_pred)
    if np.sum(~idx) > 0:
        print("Warning: %d NaN values found and excluded" % np.sum(~idx))
    return sklm.mean_absolute_error(y_true[idx], y_pred[idx])
