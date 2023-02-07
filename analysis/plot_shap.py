import os
import sys
import numpy as np
import pickle
import argparse
import shap
from skopt import dump, load
from copy import deepcopy
import matplotlib.pyplot as plt

sys.path.append('..')
sys.path.append('../scripts')
import argus_thresholds as arth
import scripts.search_params
from analysis_utils import subjects, feature_renaming


def find_model_fname(patient, artifact_dir):
    """
    Gets filename of trained model associated with the patient's test split
    
    Parameters
    ----------
    patient : str
        ID of test subject
    artifact_dir : str
        Path to directory containing saved model artifacts
        
    Returns
    -------
    fname : str
        File name of trained model
    """
    
    for fname in os.listdir(artifact_dir):
        if fname.endswith('{}.pkl'.format(patient)):
            return fname
    return None

def get_shap_vals(patients, artifact_dir, data_dir, feature_names=None):
    """
    Get shap values associated with a model specification
    
    Parameters
    ----------
    patients : list(str)
        list of patient IDs used in training/evaluation of LOSO models
    artifact_dir : str
        Path to directory containing saved model artifacts
    data_dir : str
        Path to directory containing preprocessed train and test data for 
        each validation split
        
    Returns
    -------
    shap_vals : shap.Explanation
        Shap value explanations (values, base_values, data) for each sample
        the was held out over the process of LOSO
    """
    shap_all = []
    for test_patient in patients:
        # For each subject, get the training and testing data 
        # and model for the associated LOSO split
        train_data = np.load(os.path.join(data_dir, '{}_train.npz'.format(test_patient)))
        test_data = np.load(os.path.join(data_dir, '{}_test.npz'.format(test_patient)))

        X_train, y_train = train_data['x'], train_data['y']
        X_test, y_test = test_data['x'], test_data['y']

        model_fname = find_model_fname(test_patient, artifact_dir)
        model_fpath = os.path.join(artifact_dir, model_fname)
        print(model_fpath)
        if 'hyperparam' in model_fname:
            model = load(model_fpath)
        else:
            with open(model_fpath, 'rb') as f:
                model = pickle.load(f)

        if feature_names is None:
            feature_names=arth.get_feat_cols(mode)
        
        # Compute shap explanations
        explainer = shap.Explainer(model.predict, X_train, feature_names=feature_names)
        shap_vals = explainer(X_test)
        shap_all.append(shap_vals)

    # Aggregate shap explanations from each subject
    shap_vals = deepcopy(shap_all[0])
    for i in range(1, len(shap_all)):
        shap_vals.values = np.vstack([shap_vals.values, shap_all[i].values])
        shap_vals.base_values = np.hstack([shap_vals.base_values, shap_all[i].base_values])
        shap_vals.data = np.vstack([shap_vals.data, shap_all[i].data])
    return shap_vals


parser = argparse.ArgumentParser()
parser.add_argument('--mode', \
                    choices=['routine', 'fitting', 'followup'], \
                    type=str, \
                    required=True, \
                    help='Feature set')
parser.add_argument('--data_dir', \
                    type=str, \
                    required=True, \
                    help='Directory containing training and testing data \
                          for each validation split')
parser.add_argument('--artifact_dir', \
                    type=str, \
                    required=True, \
                    help='Directory containing fitted model files')
parser.add_argument('--output_fname', \
                    type=str, \
                    required=True, \
                    help='Output filename for saving figure')
args = parser.parse_args()

if __name__ == '__main__':
    mode = args.mode
    data_dir = args.data_dir
    artifact_dir =  args.artifact_dir
    output_fname = args.output_fname

    feature_names = arth.get_feat_cols(mode)
    feature_names = [feature_renaming[feat_name] for feat_name in feature_names]

    shap_vals = get_shap_vals(subjects, artifact_dir, data_dir, feature_names=feature_names)

    use_log_scale = True
    if np.max(np.abs(shap_vals.values.flatten())) < 20:
        use_log_scale = False
    shap.summary_plot(shap_vals, max_display=10, plot_size=(10, 6), show=False, use_log_scale=use_log_scale)

    fontsize=12
    fig, ax = plt.gcf(), plt.gca()
    ax.tick_params(axis='x', which='major', labelsize=fontsize, labelrotation=45)
    ax.tick_params(axis='x', which='minor', labelsize=fontsize, labelrotation=45)
    ax.tick_params(axis='y', which='major', labelsize=fontsize)
    ax.tick_params(axis='y', which='minor', labelsize=fontsize)
    ax.set_xlabel('SHAP Value (Impact on Model Output)', fontdict={"size":fontsize})
    plt.savefig(output_fname, transparent=True)
    
    # Plot dependencies between SubjectAge and Impedance2Height
    subject_age_idx = 2
    imp2height_idx = 16
    shap.dependence_plot(subject_age_idx, shap_vals.values, shap_vals.data, feature_names=feature_names, \
                         interaction_index=imp2height_idx, show=False)
    fig, ax = plt.gcf(), plt.gca()
    ax.set_xlabel('SubjectAge (Normalized)')
    plt.tight_layout()
    plt.show()