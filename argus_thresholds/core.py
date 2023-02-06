from __future__ import absolute_import, division, print_function
import six
import numpy as np
import pandas as pd
import scipy.signal
import os.path as osp
from glob import glob
import xlrd
import sklearn.linear_model as skllm

from .utils import days_btw_cols, years_btw_cols, str2date, str2timestamp
from .feat_eng import *


__all__ = ["load_subjects", "load_data", "encode_onehot", "preprocess_data",
           "elim_greedy", "elim_lasso", "filter_min_samples", "get_feat_cols",
           "remove_first_measured_row"]


def load_subjects(csvfile):
    """Loads subject data from an CSV spreadsheet

    Assumes that there is a column called 'subject'.

    Parameters
    ----------
    csvfile : str
        Path to CSV file

    Returns
    -------
    df : pd.DataFrame
        A DataFrame with all subject data

    """
    if not isinstance(csvfile, six.string_types):
        raise TypeError("`csvfile` must be a string.")
    df = pd.read_csv(csvfile)
    return df.set_index('subject')


def fw_to_float(x):
    """Converts firmware (FW) version to float"""
    if x:
        fw = x.replace(':', '')
        if fw == '.A2E10.1':
            fw = '2.2.2.7'
        fw = fw.split('.')
        fw = '%d.%d%d%02d' % tuple([int(f) for f in fw])
        return float(fw)
    else:
        return x


def load_data(folder, subjects='all', sfile='subjects-2020.csv', verbose=True):
    """Loads threshold data from an Excel file

    Assumes that subject data can be found in a CSV file `sfile` in `folder`,
    and that for each subject listed therein, there is an Excel spreadsheet
    named after the subject ID that contains all the threshold data.

    Parameters
    ----------
    folder : str
        Path to data folder
    subjects : list or 'all', optional, default: 'all'
        List of subjects for which to load threshold data.
    sfile : str
        Name of subject data file within `folder`
    verbose : bool, optional, default: True
        If True, prints status messages along the way.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame with the threshold data

    """
    if not isinstance(folder, six.string_types):
        raise TypeError("`folder` must be a string.")
    if not isinstance(subjects, (list, np.ndarray, six.string_types)):
        raise TypeError("`subjects` must be a list or 'all'.")
    if not isinstance(sfile, six.string_types):
        raise TypeError("`sfile` must be a string.")

    sheet_th = 'Single Threshold Information'
    sheet_imp = 'Impedance Summary'

    subject_df = load_subjects(osp.join(folder, sfile))
    for col in ['notes', 'Unnamed: 14', 'Unnamed: 15']:
        if col in subject_df.columns:
            subject_df.drop(columns=col, inplace=True)

    # Subject data can be found in the 'subject' field. For every subject,
    # read the corresponding threshold data and augment it with subject data:
    dfs = []
    for subject, sdata in subject_df.iterrows():
        if isinstance(subjects, (list, np.ndarray, pd.Series)):
            if subject not in subjects:
                continue
        try:
            files = glob(osp.join(folder, '*%s*.xls' % subject))
            df = pd.read_excel(files[0], sheet_name=sheet_th)
            df_imp = pd.read_excel(files[0], sheet_name=sheet_imp)
        except (IndexError, xlrd.XLRDError) as e:
            if verbose:
                print('%s: %s' % (subject, e))
            continue

        # Add subject-specific values from subject file:
        # Add implant site:
        df['ImplantSite'] = int(subject.split('-')[0])
        df['SubjectSiteNum'] = int(subject.split('-')[1])

        # Add impedances from `sheet_imp`:
        df_imp_mean = df_imp.groupby('Test Date(in "yyyymmdd" format)').agg(
            {ele_label: lambda x: np.mean(x)
             for ele_label in df_imp.columns[2:]})

        df_x = df.merge(df_imp_mean, left_on='TestDate (in "yyyymmdd" format)',
                        right_on='Test Date(in "yyyymmdd" format)', how='left')

        df_x['Impedance (kΩ)'] = np.nan
        idx = pd.Series(df_x['ElectrodeLabel'])
        df_x['Impedance (kΩ)'] = df_x.lookup(idx.index, idx.values)
        df_x.drop(columns=[col for col in df_imp.columns[2:]], inplace=True)

        df_merge = df_x.merge(subject_df, left_on='PatientID',
                              right_on='subject', how='left')

        # Rename columns for simplicity/consistency:#
        rename = {
            'TestDate (in "yyyymmdd" format)': 'TestDate (YYYYmmdd)',
            'Maximum Current (µA)': 'MaxCurrent (µA)',
            'Inter-Phase Gap (ms)': 'InterPhaseGap (ms)',
            'CDL (mC/cm2)': 'ChargeDensityLimit (mC/cm2)',
            'Firmware Version': 'FirmwareVersion',
            'FalsePostiveRate': 'FalsePositiveRate',  # fix typo
            'gender': 'SubjectGender',
            'eye': 'ImplantEye'
        }
        df_merge.rename(mapper=rename, axis='columns', inplace=True)

        # Add firmware version as float:
        df_merge['FirmwareVersion (float)'] = np.nan
        df_merge['FirmwareVersion (float)'] = df_merge['FirmwareVersion'].apply(
            lambda x: fw_to_float(x))

        # Convert test date into more of a UNIX time stamp:
        df['TestDate (timestamp)'] = np.nan
        df_merge['TestDate (timestamp)'] = df_merge['TestDate (YYYYmmdd)'].apply(
            lambda x: str2timestamp(x, format="%Y%m%d"))

        # Calculate subject age at each data recording session:
        age = days_btw_cols(df_merge['TestDate (YYYYmmdd)'],
                            df_merge['birth_date'],
                            fmt_now="%Y%m%d")
        df_merge['SubjectAge (days)'] = pd.Series(age, index=df_merge.index)

        # Calculate age at diagnosis:
        diag = years_btw_cols(df_merge['blind_date'], df_merge['birth_date'])
        df_merge['SubjectAgeAtDiagnosis (years)'] = pd.Series(
            diag, index=df_merge.index
        )

        # Calculate age at surgery:
        surg = years_btw_cols(df_merge['surgery_date'], df_merge['birth_date'])
        df_merge['SubjectAgeAtSurgery (years)'] = pd.Series(
            surg, index=df_merge.index
        )

        # Calculate days since surgery for every data recording session:
        diff = days_btw_cols(df_merge['TestDate (YYYYmmdd)'],
                             df_merge['surgery_date'],
                             fmt_now="%Y%m%d")
        df_merge['SubjectTimePostOp (days)'] = pd.Series(diff,
                                                         index=df_merge.index)

        # Calculate years blind for every data recording session:
        diff = days_btw_cols(df_merge['TestDate (YYYYmmdd)'],
                             df_merge['blind_date'],
                             fmt_now="%Y%m%d")
        df_merge['SubjectTimeBlind (days)'] = pd.Series(diff,
                                                        index=df_merge.index)

        # these cols could be kept here and then dropped when preprocessing
        # data
        df_merge.drop(columns=['source', 'site', 'birth_date', 'blind_date',
                               'surgery_date', 'FirmwareVersion'],
                      inplace=True)

        dfs.append(df_merge)
    Xy = pd.concat(dfs, ignore_index=True)
    assert np.all(Xy.dropna()['SubjectTimePostOp (days)'] >= 0)
    assert np.all(Xy.dropna()['SubjectTimeBlind (days)'] > 0)
    assert np.all(Xy.dropna()['SubjectAge (days)'] > 0)
    return Xy


def encode_onehot(dff, columns):
    df = dff.copy()
    if isinstance(columns, six.string_types):
        columns = [columns]
    for column in columns:
        for val in df[column].unique():
            # Create a new column for every unique value. For example,
            # 'ImplantEye' might have values 'LE' and 'RE'. Here we
            # create two new columns, 'ImplantEye_LE' and 'ImplantEye_RE':
            colsplit = column.split(' ')
            if len(colsplit) == 1:
                newcolumn = '%s_%s' % (column, val)
            else:
                # If the feature's units are given in parentheses, add
                # that to the new name (e.g., 'ImplantEye_LE (eye)')
                newcolumn = '%s_%s %s' % (colsplit[0], val, colsplit[1])
            df[newcolumn] = 0
            df.loc[df[column] == val, newcolumn] = 1
    df.drop(columns=columns, inplace=True)
    return df


def remove_dead_electrodes(df, th_col='Thresholds (µA)'):
    return df[(df[th_col] > 0) & (df[th_col] < 999)].copy()


def remove_first_measured_row(dff):
    df = dff.copy()

    # Get the index of the first measured threshold:
    groupby = df.sort_values(by=['TestDate (timestamp)']).groupby(
        ['PatientID', 'ElectrodeLabel'])

    idx = groupby['TestDate (timestamp)'].apply(lambda x: x.idxmin()).values
    print("before drop last", df.shape)
    df.drop(index=idx, inplace=True)
    print("after drop last", df.shape)
    return df


def remove_zero_impedance(df, imp_col='Impedance (kΩ)'):
    return df[df[imp_col] > 0].copy()


def get_outliers_chebyshev(data, p1, p2):
    data_1 = np.array(data)
    k_1 = 1/np.sqrt(p1)
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)
    odv1_l = mean_1 - (k_1*std_1)
    odv1_u = mean_1 + (k_1*std_1)
    outlier_mask_1 = (data_1 < odv1_l) | (data_1 > odv1_u)
    
    data_2 = np.array(data_1)[~outlier_mask_1]
    k_2 = 1/np.sqrt(p2)
    mean_2 = np.mean(data_2)
    std_2 = np.std(data_2)
    odv2_l = mean_2 - (k_2*std_2)
    odv2_u = mean_2 + (k_2*std_2)
    outlier_mask = (data_1 < odv2_l) | (data_1 > odv2_u)
    return outlier_mask


def remove_outliers_chebyshev(dff, p1, p2, detrend=True):
    df = dff.copy()
    df['Outlier'] = 0
    for patient_id in df['PatientID'].unique():
        patient_df = df[df['PatientID'] == patient_id]
        for electrode_label in patient_df['ElectrodeLabel'].unique():
            patient_electrode_idx = patient_df[patient_df['ElectrodeLabel'] == electrode_label].index
            thresholds = df.loc[patient_electrode_idx,'Thresholds (µA)'].to_numpy()
            if detrend:
                thresholds = scipy.signal.detrend(thresholds)
            outlier_mask = get_outliers_chebyshev(thresholds, p1, p2)
            outlier_mask = outlier_mask.astype(np.uint8)
            df.loc[patient_electrode_idx, 'Outlier'] = outlier_mask
    df = df[df['Outlier'] == 0]
    #df = df.drop(['Outlier'], axis=1)
    return df
    

def preprocess_data(dff, ignore_dead_electrodes=True, remove_outliers=True, threshold_scaling=None, remove_first_row=False):
    
    # One-hot encode categorical features:
    df = encode_onehot(dff, ['SubjectGender', 'ImplantEye', 'ImplantSite'])
    
    # Drop redundant categorical features:
    df.drop(columns=['ImplantEye_LE', 'SubjectGender_M'], inplace=True)

    # Add electrode location and RGC density from fundus images:
    df = add_feat_fundus(df)

    # Add mean impedance:
    df = add_feat_mean_impedance(df)

    # Impedance to threshold (deBalthasar et al., 2008):
    df = add_feat_imp2th(df)

    # Impedance to electrode-retina distance (deBalthasar et al., 2008):
    df = add_feat_imp2height(df)

    # Impedance to height to threshold (deBalthasar et al., 2008):
    df = add_feat_imp2height2th(df)
    
    # Fraction of dead electrodes on a given date for the subject.
    # This must be put before add_feat_first_visit and add_feat_last_visit.
    df = add_feat_dead_electrodes(df)

    # Add data from first visit:
    df = add_feat_first_visit(df)
    
    # Remove all electrodes with threshold==999:
    if ignore_dead_electrodes:
        df = remove_dead_electrodes(df)

    # Remove electrodes with zero impedance measurement
    df = remove_zero_impedance(df)
        
    if remove_outliers:
        # Remove outliers accoring to chebyshev outlier rejection
        df = remove_outliers_chebyshev(df, 0.2, 0.1)
    
    # Scale electrode thresholds
    if threshold_scaling is not None:
        df = scale_electrode_thresholds(df, threshold_scaling)
        df['Thresholds (µA)'] = df['Thresholds (scaled)']
    
    # Add amount of time since first meaurement was performed
    df = add_feat_time_since_first_measurement(df)
    
    # Add amount of time elapsed since each electrode was previously measured
    df = add_feat_time_since_last_electrode_measurement(df)

    # Add last threshold meaurement
    df = add_feat_last_thresholds(df)

    # Add last impedance measurement
    df = add_feat_last_impedance(df)

    # Remove first visit rows, as these thresholds are used as features elsewhere
    if remove_first_row:
        df = remove_first_measured_row(df)

    # Drop all the clutter:
    dropcols = ['HotValueCorrected Thresholds (µA) ', 'ElectrodeIndex',
                'PulseWidth (ms)', 'Frequency (Hz)', 'Duration (ms)',
                'InterPhaseGap (ms)', 'FalsePositiveRate',
                'ChargeDensityLimit (mC/cm2)', 'MaxCurrent (µA)',
                'TestDate (YYYYmmdd)']
    df = df.drop(columns=dropcols)
    
    df = df.replace([np.inf, -np.inf], np.nan)

    return df.reset_index(drop=True)


def elim_greedy(Xy, pred_col='Thresholds (µA)', verbose=True):
    """Orders features how they should be selected to minimize correlation"""
    if Xy.shape[0] == 0:
        return []

    dropcols = [pred_col]
    max_iter = len(Xy.drop(columns=dropcols).columns)
    iter = 0
    while len(Xy.drop(columns=dropcols).columns) > 0:
        if iter > max_iter:
            print("Maximum number of iterations reached.")
            break
        if verbose:
            print(Xy.drop(columns=dropcols).columns)
        # Correlation matrix:
        corr = Xy.drop(columns=dropcols).corr().abs()
        corr.dropna(how='all', axis=0, inplace=True)
        corr.dropna(how='all', axis=1, inplace=True)
        if len(corr) == 1:
            dropcols.append(corr.columns[0])
            break

        # Pairs:
        pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        pairs = pairs.stack().sort_values(ascending=False)
        for (key1, key2), _ in pairs.iteritems():
            if verbose:
                print(key1, key2)
            corr1 = Xy[key1].corr(Xy[pred_col])
            corr2 = Xy[key2].corr(Xy[pred_col])
            if corr1 < corr2:
                dropcols.append(key1)
            else:
                dropcols.append(key2)
                # We just want the highest correlated pair:
            break
        iter += 1
    return dropcols[1:][::-1]


def elim_lasso(Xy, n_steps=21, pred_col='Thresholds (µA)'):
    if Xy.shape[0] == 0:
        return []

    dropcols = ['PatientID', 'ElectrodeLabel'] + [pred_col]
    featcols = Xy.drop(columns=dropcols).columns
    alphas = np.linspace(0, 1, n_steps)
    lasso_feats = []
    for alpha in alphas:
        if np.isclose(alpha, 0):
            # Lasso is unstable for small alphas, use linreg:
            model = skllm.LinearRegression(normalize=True)
        else:
            model = skllm.Lasso(alpha=alpha, normalize=True)
        model.fit(Xy[featcols], Xy[pred_col])
        lasso_feats.append(featcols[~np.isclose(model.coef_, 0)])
    select_feats = []
    for feat in lasso_feats[::-1]:
        for f in feat:
            if f not in select_feats:
                select_feats.append(f)
    return select_feats


def filter_min_samples(Xy, n_min_samples, groupby='PatientID'):
    """Removes subjects with not enough samples

    Parameters
    ----------
    Xy : pd.DataFrame
        Combined feature and target matrices
    n_min_samples : int
        Remove subjects with less than `n_min_samples` data points.
    groupby: str or list of str
        DataFrame will be grouped by this string or list of strings.

    Examples
    --------
    Retain subjects with more than 5 data points:
        filter_min_samples(Xy, 5, groupby='PatientID')
    """
    if not isinstance(Xy, pd.DataFrame):
        raise TypeError('`Xy` must be a Pandas DataFrame.')
    if not isinstance(n_min_samples, int):
        raise TypeError("`n_min_samples` must be an integer.")

    # Remove subjects with less than `n_min_samples` entries:
    Xy = Xy.groupby(groupby).filter(
        lambda x: x[groupby].count() >= n_min_samples
    )
    return Xy


def get_feat_cols(mode):
    """Returns the feature columns for a given mode

    Parameters
    ----------
    mode : str
        Modes:
        - "routine": routinely collected clinical parameters and 
                     parameters estimates from implant imagery
        - "fitting": "clinical" + system fitting features (first visit)
        - "followup": "fitting" + features from previous visits
    """
    
    routine_features = ['Impedance (kΩ)', 'ImpedanceCV (std/mu)', \
                        'SubjectAge (days)', 'SubjectTimePostOp (days)', \
                        'SubjectAgeAtDiagnosis (years)', 'SubjectTimeBlind (days)', \
                        'SubjectAgeAtSurgery (years)', 'ElectrodeLocRho (µm)', \
                        'ElectrodeLocTheta (rad)', 'ImplantMeanLocRho (µm)', \
                        'ImplantMeanLocTheta (rad)', 'ImplantMeanRot (rad)', \
                        'OpticDiscLocX (µm)', 'OpticDiscLocY (µm)', \
                        'RGCDensity (cells/deg2)', 'Impedances2Thresholds (µA)', \
                        'Impedances2Height (µm)', 'Impedances2Heights2Thresholds (µA)']

    fitting_features = ['FirstThresholds (µA)', 'FirstImpedance (kΩ)', \
                        'FirstFalsePositiveRate', 'FirstElectrodesDead (frac)', \
                        'FirstChargeDensityLimit (mC/cm2)', 'FirstMaxCurrent (µA)', \
                        'TimeSinceFirstMeasurement (days)']
            
    followup_features = ['TimeSinceLastElectrodeMeasurement (days)', \
                         'LastThresholds (µA)', 'LastImpedance (kΩ)']

    # Define a mapping from mode string to list of columns:
    mapping = {
        'routine': routine_features,
        'fitting': routine_features + fitting_features,
        'followup': routine_features + fitting_features + followup_features,
    }
    if not isinstance(mode, str):
        raise TypeError("'mode' must be a string")
    if mode not in mapping.keys():
        raise ValueError("'mode' must be 'routine', 'fitting', or 'followup'.")
    return mapping[mode]
