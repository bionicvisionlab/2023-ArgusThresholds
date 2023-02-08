import os
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from skopt import load

# 12 analyzed subjects
subjects = ['12-001', '12-004', '12-005', '14-001', \
            '17-002', '51-001', '51-003', '51-009', \
            '52-001', '52-003', '61-004', '71-002']

# Feature name reformatting for plotting
feature_renaming = {'SubjectAge (days)': 'SubjectAge',
                    'SubjectTimeBlind (days)': 'SubjectTimeBlind',
                    'SubjectAgeAtDiagnosis (years)': 'SubjectAgeAtDiagnosis',
                    'SubjectAgeAtSurgery (years)': 'SubjectAgeAtSurgery',
                    'SubjectTimePostOp (days)': 'ImplantTime',
                    'Impedance (kΩ)': 'Impedance',
                    'ImpedanceCV (std/mu)': 'ImpedanceCV',
                    'ElectrodeLocRho (µm)': 'ElectrodeLocRho',
                    'ElectrodeLocTheta (rad)': 'ElectrodeLocTheta',
                    'ImplantMeanLocRho (µm)': 'ImplantMeanLocRho',
                    'ImplantMeanLocTheta (rad)': 'ImplantMeanLocTheta',
                    'ImplantMeanRot (rad)': 'ImplantMeanRot',
                    'OpticDiscLocX (µm)': 'OpticDiscLocX',
                    'OpticDiscLocY (µm)': 'OpticDiscLocY',
                    'RGCDensity (cells/deg2)': 'RGCDensity',
                    'Impedances2Thresholds (µA)': 'Impedances2Thresholds',
                    'Impedances2Height (µm)': 'Impedances2Height',
                    'Impedances2Heights2Thresholds (µA)': 'Impedances2Heights2Thresholds',
                    'FirstImpedance (kΩ)': 'FirstImpedance',
                    'FirstThresholds (µA)': 'FirstThresholds',
                    'FirstMaxCurrent (µA)': 'FirstMaxCurrent',
                    'FirstChargeDensityLimit (mC/cm2)': 'FirstChargeDensityLimit',
                    'FirstElectrodesDead (frac)': 'FirstDeactivationRate',
                    'FirstFalsePositiveRate': 'FirstFalsePositiveRate',
                    'TimeSinceFirstMeasurement (days)': 'TimeSinceFirstMeasurement',
                    'LastImpedance (kΩ)': 'LastImpedance',
                    'LastThresholds (µA)': 'LastThresholds',
                    'TimeSinceLastElectrodeMeasurement (days)': 'TimeSinceLastMeasurement'}

def get_model_results(fpath):
    """
    Get regression result data for each subject
    
    Parameters
    ----------
    fpath : str
        Path to saved model artifact file
    
    Returns
    -------
    X_tests : dict
        Dictionary mapping subject identifier to subject test data
    y_tests : dict
        Dictionary mapping subject identifier to ground truth perceptural thresholds
    y_hats : dict
        Dictionary mapping subject identifier to regression model predictions
    """
    X_tests = {}
    y_tests = {}
    y_hats = {}
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
        
    for i in range(len(data['held-out-patients'])):
        subject = data['held-out-patients'][i]
        X_tests[subject] = np.array(data['X_test'][i])
        y_tests[subject] = np.array(data['y_test'][i])
        y_hats[subject] = np.array(data['y_pred'][i])
    return X_tests, y_tests, y_hats


def get_adjusted_r2(r2, n_samples, n_predictors):
    """
    Compute adjusted r2 score
    
    Parameters
    ----------
    r2 : float
        r2 score
    n_samples : int
        number of test samples
    n_predictors : int
        number of predictors in test data
        
    Returns
    -------
    r2_adjusted : float
        Adjusted r2 score
    """
    r2_adjusted = 1-(((1-r2)*(n_samples-1))/(n_samples-n_predictors-1))
    return r2_adjusted


def get_auc(artifact_dir, results_fname, eval_subjects):
    """
    Compute AUC scores for subjects in eval_subjects
    
    Parameters
    ----------
    artifact_dir : str
        Path to directory with model artifacts
    results_fname : str
        Path of saved results file
    eval_subjects : list(str)
        List of subject IDs to evaluate
        
    Returns
    -------
    auc : float
        AUC of data aggregated from eval_subjects
    """
    with open(results_fname, 'rb') as f:
        results_data = pickle.load(f)
        
    y_test_all, y_hat_all = [], []
    for i, patient in enumerate(results_data['held-out-patients']):
        if patient not in eval_subjects:
            continue
        X_test = results_data['X_test'][i]
        y_test = results_data['y_test'][i]
        
        model_fname = None
        for fn in os.listdir(artifact_dir):
            if fn.endswith('{}.pkl'.format(patient)):
                model_fname = fn
                break
        if model_fname is None: return None
        model_fname = os.path.join(artifact_dir, model_fname)
    
        model = load(model_fname)
        y_hat = model.predict_proba(X_test)[:,1]
        
        y_hat_all.append(y_hat)
        y_test_all.append(y_test)
        
    y_hat_all = np.hstack(y_hat_all)
    y_test_all = np.hstack(y_test_all)
    return roc_auc_score(y_test_all, y_hat_all)