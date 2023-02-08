import numpy as np
import pandas as pd
import pickle
import argparse
from sklearn.metrics import mean_squared_error
from analysis_utils import subjects, get_model_results


def compute_feve(mse, var, sigma):
    """
    Compute feve from MSE, response variance, and expected stimulus variance
    
    Parameters
    ----------
    mse : float
        Predicition mean squared error
    var : float
        Response variance
    sigma : float
        Expected stimulus variance
        
    Returns
    -------
    feve : float
        fraction of explainable variance explained
    """
    return 1 - ((mse-sigma)/(var-sigma))


def group_timestamps(sorted_timestamps, max_dist=14, min_group_size=2):
    """
    Groups timestamps within a difference of max_dist from eachother
    
    Parameters
    ----------
    sorted_timestamps : list
        List of sorted event timestamps
    max_dist : int
        Maximum distance between timestamps to be considered \
        from the same group
    min_group_size : int
        Minimim size of timestamp grouping to consider
        
    Returns
    -------
    groups : list(list)
        List of grouped timestamps
    inds : list(list)
        List of grouped timestamp indices
    """
    group, groups = [], []
    group_inds, inds = [], []
    for i in range(len(sorted_timestamps)):
        current_ts = sorted_timestamps[i]
        group = [current_ts]
        group_inds = [i]
        for j in range(i+1, len(sorted_timestamps)):
            future_ts = sorted_timestamps[j]
            if future_ts - current_ts <= max_dist:
                group.append(future_ts)
                group_inds.append(j)
            else:
                break
        if len(group) >= min_group_size:
            groups.append(group)
            inds.append(group_inds)
    return groups, inds


parser = argparse.ArgumentParser()
parser.add_argument('--data_file', \
                    type=str, \
                    required=True, \
                    help='Path to file with feature data for each subject')
parser.add_argument('--results_file', \
                    type=str, \
                    required=True, \
                    help='Path to saved model artifact file')
parser.add_argument('--max_dists', \
                    nargs='+',
                    type=int,
                    required=True, \
                    help='Maximum number of days between measurements to treat as same stimulus')
args = parser.parse_args()
if __name__ == '__main__':
    data_file = args.data_file
    results_file = args.results_file
    max_dists = args.max_dists
    
    thresholds_df = pd.read_csv(data_file)
    X_tests, y_tests, y_hats = get_model_results(results_file)
    y_test_all = np.hstack([y_tests[subject] for subject in subjects])
    y_hat_all = np.hstack([y_hats[subject] for subject in subjects])
    
    feve_data = []
    for max_dist in max_dists:
        # Compute subject-specific feve variables
        subject_mses = []
        subject_variances = []
        subject_sigmas = []
        subject_response_variances = {test_subject: [] for test_subject in subjects}
        for i, test_subject in enumerate(subjects):
            subject_feve_df = thresholds_df[thresholds_df['PatientID'] == test_subject].copy()
            electrodes = subject_feve_df.ElectrodeLabel.unique()
            # Get compute sigma from repeat measurements within max_dist dats from same electrode
            for electrode in electrodes:
                electrode_df = subject_feve_df[subject_feve_df['ElectrodeLabel'] == electrode].copy()
                electrode_df = electrode_df[['SubjectTimePostOp (days)', 'Thresholds']]
                electrode_df = electrode_df.sort_values('SubjectTimePostOp (days)').to_numpy()
                recording_times = np.array(electrode_df[:,0])
                _, idx_groups = group_timestamps(recording_times, max_dist=max_dist)
                for inds in idx_groups:
                    # Repeat measurements for same electrode
                    inds = np.array(inds)
                    group_thresholds = electrode_df[inds, 1]
                    subject_response_variances[test_subject].append(np.var(group_thresholds))
            subject_mses.append(mean_squared_error(y_tests[test_subject], y_hats[test_subject]))
            subject_variances.append(np.var(subject_feve_df['Thresholds'].to_numpy()))
            subject_sigmas.append(np.mean(subject_response_variances[test_subject]))
            
        subject_feves = np.zeros(len(subjects))
        for i in range(len(subjects)):
            subject_feves[i] = compute_feve(subject_mses[i], subject_variances[i], subject_sigmas[i])
            
        # Compute global feve variables
        global_feve_df = thresholds_df.loc[thresholds_df['PatientID'].isin(subjects)].copy()
        global_variance = np.var(global_feve_df['Thresholds'].to_numpy())
        global_mse = mean_squared_error(y_test_all, y_hat_all)
        global_sigma = np.mean(np.hstack([r_var for r_var in subject_response_variances.values()]))
        global_feve = compute_feve(global_mse, global_variance, global_sigma)

        print('Global FEVE_{}: {:.4f}'.format(max_dist, global_feve))
        print('Local FEVE_{} (mean +/- std dev): {:.4f} \pm {:.4f}'\
              .format(max_dist, np.nanmean(subject_feves), np.nanstd(subject_feves)))
        print()