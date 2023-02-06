import numpy as np
import pandas as pd
from .cv import cross_val_predict
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from sklearn.metrics import r2_score

from .utils import print_boundary_warning
from .metrics import calc_r2


__all__ = ['predict_fit', 'predict_cv', 'tune_params',
           'fine_tune_params']


def predict_fit(X, y, model, pparams, return_model=False):
    """Fit the model to the entire dataset

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Vector of target values
    model : sklearn estimator
        A scikit-learn estimator with ``fit`` and ``predict`` methods
    pparams : dict
        Dictionary of model parameters. For early stopping, add a dictionary
        entry 'fit_params': {'early_stopping_rounds': N}
    return_model : bool
        If True, returns the fitted model as well

    Returns
    -------
    y_pred : pd.Series
        Vector of predictions
    model : sklearn estimator
        If ``return_model`` is True, returns the fitted model
    """
    params = pparams.copy()
    if 'fit_params' in params.keys():
        fit_params = params['fit_params'].copy()
        for key in ['early_stopping_rounds', 'eval_split']:
            if key in fit_params.keys():
                del fit_params[key]
                print("Ignoring %s for fit" % key)
        del params['fit_params']
    else:
        fit_params = {}
    m = model(**params)
    m.fit(X, y, **fit_params)
    if return_model:
        return m.predict(X), m
    return m.predict(X)


def predict_cv(X, y, model, pparams, groups):
    """Perform leave-one-subject-out cross-validation

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Vector of target values
    model : sklearn estimator
        A scikit-learn estimator with ``fit`` and ``predict`` methods
    pparams : dict
        Dictionary of model parameters. For early stopping, add a dictionary
        entry 'fit_params': {'early_stopping_rounds': N}

    Returns
    -------
    y_pred : pd.Series
        Vector of predictions
    """
    params = pparams.copy()
    if 'fit_params' in params.keys():
        fit_params = params['fit_params'].copy()
        del params['fit_params']
    else:
        fit_params = None
    m = model(**params)
    return cross_val_predict(m, X, y=y, groups=groups,
                             fit_params=fit_params,
                             cv=LeaveOneGroupOut())


def tune_params(XX, yy, model, ssearch_params, iinit_params, groups,
                score=calc_r2, verbose=True):
    X = XX.copy()
    y = yy.copy()
    search_params = ssearch_params.copy()
    init_params = iinit_params.copy()

    # If a search parameter is also in the init grid, remove it
    # from there:
    for param in search_params.keys():
        if param in init_params:
            del init_params[param]

    # Search the grid and record the scores:
    results = []
    for params in ParameterGrid(search_params):
        if groups is None:
            y_pred = predict_fit(X, y, model, {**params, **init_params})
        else:
            y_pred = predict_cv(X, y, model, {**params, **init_params}, groups)
        params['score'] = score(y, y_pred)
        results.append(params)
        if verbose:
            print('-', params)
    results = pd.DataFrame(results)

    # Find the best score:
    best_idx = results['score'].idxmax()
    if verbose:
        print(results.loc[best_idx, list(search_params.keys()) + ['score']])

    # Replace init_params with the new values:
    for key in search_params.keys():
        init_params[key] = results.loc[best_idx, key]
        print_boundary_warning(key, init_params[key], search_params[key])

    return init_params, results


def fine_tune_params(XX, yy, model, ssearch_params, iinit_params, groups,
                     fine_tune, fine_steps=11, verbose=True):
    # Coarse-grained search:
    init_params, results = tune_params(XX, yy, model, ssearch_params,
                                       iinit_params, groups, verbose=verbose)

    # Need to fine-tune (e.g. when we ran a logspace grid)
    fine_search_params = {}
    for key in fine_tune:
        idx = ((init_params[key] - ssearch_params[key]) ** 2).argmin()
        lower_bound = ssearch_params[key][max(0, idx - 1)]
        upper_bound = ssearch_params[key][min(len(ssearch_params[key]) - 1,
                                              idx + 1)]
        fine_search_params[key] = np.unique(np.linspace(
            lower_bound, upper_bound, num=fine_steps,
            dtype=ssearch_params[key].dtype
        ))

    # Fine-grained search:
    init_params, fine_results = tune_params(XX, yy, model, fine_search_params,
                                            init_params, groups,
                                            verbose=verbose)
    results = pd.concat((results, fine_results), ignore_index=True)

    return init_params, results.reset_index()
