#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('..')
import os
import time
import copy
import argparse
import numpy as np
import pandas as pd
import warnings
import pickle
from skopt import BayesSearchCV, dump, load
from search_params import xgb_base_model, xgb_search_params, elasticnet_base_model, elasticnet_search_params
import argus_thresholds as arth

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', \
                    type=str, \
                    required=True, \
                    help='Path to preprocessed data')
parser.add_argument('--mode', \
                    choices=['routine', 'fitting', 'followup'], \
                    type=str, \
                    required=True, \
                    help='Feature set to use for model fitting')
parser.add_argument('--model', \
                    choices=['xgb', 'elasticnet'], \
                    type=str, \
                    required=True, \
                    help='Model to use for regression fitting/evaluation')
parser.add_argument('--standardize', \
                    default=False, \
                    action='store_true', \
                    help='Standardize feature values')
parser.add_argument('--paramsearch', \
                    default=False, \
                    action='store_true', \
                    help='Perform hyperparameter search for classifier')
parser.add_argument('--outpath', \
                    nargs='?', \
                    type=str, \
                    help='Path to directory where output and results will be saved')
args = parser.parse_args()


def main():
    # Mode string
    datapath = args.datapath
    mode = args.mode
    model_type = args.model
    standardize = args.standardize
    paramsearch = args.paramsearch
    outpath = args.outpath
    print('datapath: ', datapath)
    print('mode: ', mode)
    print('model: ', model_type)
    print('standardize: ', standardize)
    print('paramsearch: ', paramsearch)
    print('outpath: ', outpath)
    
    if outpath is None:
        outpath = '%s-experiment' % model_type
        
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    else:
        outpath += '-%s' % str(time.time())
        print('Output path already exists.  Using %s instead' % outpath)
        os.mkdir(outpath)
        
    results_file = 'results-%s-%s.pkl' % (model_type, mode)
    results_file = os.path.join(outpath, results_file)
        
    Xy = pd.read_csv(datapath)
    assert len(Xy) > 0, 'Dataframe must have more than 0 samples'
    
    patients = Xy.PatientID.unique()
        
    feature_columns = arth.get_feat_cols(mode)
    label_column = ['Thresholds']

    results = {}
    all_X_test = []
    all_y_test = []
    all_y_pred = []
    patient_r2s = []
    for fold_i, patient in enumerate(patients):
        test_patients = [patient]
        train_patients = [pid for pid in patients if pid not in test_patients]
        
        X_train, y_train, patient_inds = arth.cv.get_fold_data(Xy, feature_columns, \
                                                               label_column, train_patients)
        loo_cv_inds = arth.cv.get_loo_cross_validation_inds(patient_inds)
        X_test, y_test, _ = arth.cv.get_fold_data(Xy, feature_columns, \
                                                  label_column, test_patients)
        
        if standardize:
            print(feature_columns)
            feature_means = np.nanmean(X_train, axis=0)
            feature_stds = np.nanstd(X_train, axis=0)
            if np.any(feature_stds == 0):
                warnings.warn('Some features have std. dev. equal to 0.' \
                              'Ignorning in standardization')
                feature_stds[feature_stds == 0] = 1
            X_train = (X_train-feature_means)/feature_stds
            X_test = (X_test-feature_means)/feature_stds
            
        if paramsearch:
            if model_type == 'xgb':
                search_params = xgb_search_params
                search_params['base_score'] = np.mean(y_train)
            else:
                # model_type is 'elasticnet':
                search_params = elasticnet_search_params

            model = BayesSearchCV(estimator=search_params['estimator'],\
                                  search_spaces=search_params['search_spaces'],
                                  n_iter=search_params['n_iter'],
                                  cv=loo_cv_inds,
                                  scoring=search_params['scoring'],
                                  refit=search_params['refit'],
                                  random_state=search_params['random_state'],
                                  n_jobs=-1, verbose=0)
            model.fit(X_train, y_train)
            dump(model, os.path.join(outpath, 'hyperparam_search_%s_%s.pkl' % (model_type, patient)))
            
        else:
            if model_type == 'xgb':
                model = xgb_base_model
            else:
                # model_type is 'elasticnet':
                model = elasticnet_base_model
                
            model.fit(X_train, y_train)
            model_fname = os.path.join(outpath, 'base_%s_%s.pkl' % (model_type, patient))
            with open(model_fname, 'wb') as f:
                pickle.dump(model, f)
            
        y_pred = model.predict(X_test)
        all_y_pred.append(y_pred)
        all_X_test.append(X_test)
        all_y_test.append(y_test)
        
        patient_r2 = arth.metrics.calc_r2(y_test, y_pred)
        patient_r2s.append(patient_r2)
        print('-'*100)
        print('\nFold %s:' % fold_i)
        print('\tNumber of test samples: %s' % X_test.shape[0])
        print('\tR2 score: %s' % patient_r2)
        print('-'*100)
        
    # Calculate aggregated
    X_test_agg = np.vstack(all_X_test)
    y_test_agg = np.hstack(all_y_test)
    y_pred_agg = np.hstack(all_y_pred)
    r2 = arth.metrics.calc_r2(y_test_agg, y_pred_agg)
    results = {'held-out-patients': patients,
               'standardize': standardize,
               'X_test': all_X_test, 'y_test': all_y_test, 'y_pred': all_y_pred,
               'loo_R2s': patient_r2s,
               'R2': r2}
    print('\nAggregated LOO results:')
    print('\tNumber of test samples: %s' % X_test_agg.shape[0])
    print('\tR2 score: %s' % r2)

    with open(results_file, 'wb') as file:
        pickle.dump(results, file)
    print('Data saved to %s' % results_file)

if __name__ == "__main__":
    main()
