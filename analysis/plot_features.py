import sys
import numpy as np
import pandas as pd
import scipy.stats
import argparse
import matplotlib.pyplot as plt

sys.path.append('..')
from argus_thresholds import get_feat_cols
from analysis_utils import subjects, feature_renaming

def get_linear_regression_line_with_slope(X, y, n_samples=1000):
    slope, intercept, r, p, _ = scipy.stats.linregress(X, y)
    x_samples = np.linspace(min(X), max(X), n_samples)
    y_samples = (x_samples*slope) + intercept
    return x_samples, y_samples, r, slope, p

parser = argparse.ArgumentParser()
parser.add_argument('--mode', \
                    choices=['routine', 'fitting', 'followup'], \
                    type=str, \
                    required=True, \
                    help='Feature set')
parser.add_argument('--datapath', \
                    type=str, \
                    required=True, \
                    help='Path to csv file containing processed data of all subjects')
parser.add_argument('--output_fname', \
                    type=str, \
                    required=True, \
                    help='Output filename for saving figure')
args = parser.parse_args()

tick_fontsize = 12
label_fontsize = 12
num_rows = 7
num_cols = 4
if __name__ == '__main__':
    mode =  args.mode
    datapath = args.datapath
    output_fname = args.output_fname
    features = get_feat_cols(mode)
    
    df = pd.read_csv(datapath)
    df = df.loc[df['PatientID'].isin(subjects)]
    df = arth.core.remove_dead_electrodes(df, th_col='Thresholds')
    threshold_data = df['Thresholds'].to_numpy()
    
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 3*num_rows), sharex=False, sharey=True)
    for ax, feature in zip(axes.ravel(), features):
        x, y = df[feature].to_numpy(), threshold_data
        x_linear, y_linear, r, slope, p = get_linear_regression_line_with_slope(x, y)
        regression_label = 'p={:.3f}'.format(p) if p >= 0.001 else 'p < 0.001'
        
        ax.scatter(x, y, color='lightblue', linewidths=1, edgecolors='steelblue', alpha=0.6)
        ax.plot(x_linear, y_linear, '--', color='k', label='m={:.3f} ({}), r={:.3f}'.format(slope, regression_label, r))
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(-4,4))
        ax.set_ylim(0, 700)
        ax.set_xlabel(feature_renaming[feature], size=label_fontsize)
        ax.set_yticks(np.arange(0, 800, 100))
        ax.tick_params(axis='x', labelsize=tick_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        ax.grid(alpha=0.6)
        ax.legend(loc='upper left')

    fig.text(-0.01, 0.5, 'Perceptual Threshold (µA)', ha='center', va='center', rotation=90, size=12)
    plt.subplots_adjust(hspace=1)
    plt.tight_layout()
    plt.savefig(output_fname, transparent=True, bbox_inches='tight')