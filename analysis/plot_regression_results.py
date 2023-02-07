import os
import sys
import numpy as np
import pandas as pd
import argparse
import pickle
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

sys.path.append('..')
from analysis_utils import subjects, get_adjusted_r2


parser = argparse.ArgumentParser()
parser.add_argument('--mode', \
                    choices=['routine', 'fitting', 'followup'], \
                    type=str, \
                    required=True, \
                    help='Feature set')
parser.add_argument('--model_type', \
                    choices=['elasticnet', 'xgb'], \
                    type=str, \
                    required=True, \
                    help='Feature set')
parser.add_argument('--artifact_dir', \
                    type=str, \
                    required=True, \
                    help='Directory containing fitted model files')
parser.add_argument('--output_fname', \
                    type=str, \
                    required=True, \
                    help='Output filename for saving figure')
args = parser.parse_args()

tick_fontsize = 12
label_fontsize = 12
title_fontsize = 12
if __name__ == '__main__':
    artifact_dir = args.artifact_dir
    mode = args.mode
    model_type = args.model_type
    output_fname = args.output_fname
    
    model_results_fname = os.path.join(artifact_dir, 'results-{}-{}.pkl'.format(model_type, mode))
    with open(model_results_fname, 'rb') as f:
        result_data = pickle.load(f)

    subject_thresholds = {}
    subject_predictions = {}
    subject_data = {}
    subjects = result_data['held-out-patients']
    for i, subject in enumerate(subjects):
        subject_thresholds[subject] = np.array(result_data['y_test'][i])
        subject_predictions[subject] = np.array(result_data['y_pred'][i])
        subject_data[subject] = np.array(result_data['X_test'][i])
    threshold_units = 'µA' if mode == 'routine' else 'scaled'

    num_rows = 3
    num_cols = 4
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 3*num_rows), \
                             sharex=False, sharey=False)
    for ax, subject in zip(axes.ravel(), sorted(subjects)):
        X_test = subject_data[subject]
        y_test, y_hat = subject_thresholds[subject], subject_predictions[subject]
        
        r2 = r2_score(y_test, y_hat)
        n_samples, n_params = X_test.shape
        r2 = get_adjusted_r2(r2, n_samples, n_params)
        
        ax.scatter(y_test, y_hat, color='lightblue', linewidths=1, edgecolors='steelblue', alpha=0.6)

        min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([min_val, max_val], [min_val, max_val], color='k', \
                label=r'$R^2_{adj}$='+'{:.3f}'.format(r2), alpha=0.4, lw=1)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        ax.set_title(subject, size=title_fontsize)
        ax.grid(alpha=0.6)
        ax.legend(loc='upper left', fontsize=tick_fontsize)

    fig.text(-0.01, 0.5, 'Predicted Perceptual Threshold ({})'.format(threshold_units), \
             ha='center', va='center', rotation=90, size=label_fontsize)
    fig.text(0.5, -0.01, 'Ground Truth Perceptual Threshold ({})'.format(threshold_units), \
             ha='center', va='center', rotation=0, size=label_fontsize)
    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    plt.savefig(output_fname, transparent=False, bbox_inches='tight')