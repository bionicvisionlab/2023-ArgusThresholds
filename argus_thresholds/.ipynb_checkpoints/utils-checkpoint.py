import numpy as np
from pandas import notnull
import time
from datetime import datetime
import six

__all__ = ['str2date', 'str2timestamp', 'years_btw_dates', 'days_btw_dates',
           'print_boundary_warning', 'icc', 'binarize_dead_electrodes']


def binarize_dead_electrodes(thresholds, dead_electrode_threshold=999):
    thresholds = np.array(thresholds)
    thresholds[thresholds != dead_electrode_threshold] = 0
    thresholds[thresholds == dead_electrode_threshold] = 1
    return thresholds

def str2date(datestr, format="%Y-%m-%d"):
    if notnull(datestr):
        try:
            if not isinstance(datestr, six.string_types):
                datestr = str(datestr)
            return datetime.strptime(datestr, format)
        except:
            return np.nan
    return np.nan


def str2timestamp(datestr, format="%Y-%m-%d"):
    try:
        return str2date(datestr, format=format).timestamp()
    except AttributeError:
        return np.nan


def years_btw_dates(date_now, date_then):
    try:
        year = date_now.year - date_then.year
        months = ((date_now.month, date_now.day)
                  < (date_then.month, date_then.day))
        return year - months
    except:
        return np.nan


def years_btw_cols(col_now, col_then, fmt_now="%Y-%m-%d", fmt_then="%Y-%m-%d"):
    """Calculates the number of years between two dates

    Parameters
    ----------
    col_now, col_then : pd.Series
        A column of a DataFrame containing the start (col_then) and end date
        (col_now)
    fmt_now, fmt_then : str
        Date format in which col_new and col_then are in

    """
    # If any values are NaN, str2date and days_btw_dates will return NaN
    return [years_btw_dates(str2date(now, format=fmt_now),
                            str2date(then, format=fmt_then))
            for now, then in zip(col_now, col_then)]


def print_boundary_warning(name, val, param_list):
    if val == param_list[0] or val == param_list[-1]:
        print("Warning,", name, "=", val,
              "is at the boundary of the search grid")


def days_btw_dates(date_now, date_then):
    """Calculate the number of days between two dates"""
    try:
        return (date_now - date_then).days
    except:
        return np.nan


def days_btw_cols(col_now, col_then, fmt_now="%Y-%m-%d", fmt_then="%Y-%m-%d"):
    """Calculates the number of days between two dates

    Parameters
    ----------
    col_now, col_then : pd.Series
        A column of a DataFrame containing the start (col_then) and end date
        (col_now)
    fmt_now, fmt_then : str
        Date format in which col_new and col_then are in

    """
    # If any values are NaN, str2date and days_btw_dates will return NaN
    return [days_btw_dates(str2date(now, format=fmt_now),
                           str2date(then, format=fmt_then))
            for now, then in zip(col_now, col_then)]


def icc(data, icc_type='icc2'):
    ''' Calculate intraclass correlation coefficient for data within
        Brain_Data class
    ICC Formulas are based on:
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
    assessing rater reliability. Psychological bulletin, 86(2), 420.
    icc1:  x_ij = mu + beta_j + w_ij
    icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij
    Code modifed from nipype algorithms.icc
    https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py
    Args:
        icc_type: type of icc to calculate (icc: voxel random effect,
                icc2: voxel and column random effect, icc3: voxel and
                column fixed effect)
    Returns:
        ICC: (np.array) intraclass correlation coefficient
    '''

    Y = np.asarray(data)
    [n, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k - 1)
    dfr = n - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))),
                                X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    MSC = SSC / dfc / n

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == 'icc1':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        # ICC = (MSR - MSRW) / (MSR + (k-1) * MSRW)
        NotImplementedError("This method isn't implemented yet.")

    elif icc_type == 'icc2':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)

    elif icc_type == 'icc3':
        # ICC(3,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error)
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE)

    return ICC
