import sys
import numpy as np
import pandas as pd
import scipy.stats
import argparse
import matplotlib.pyplot as plt

sys.path.append('..')
import argus_thresholds as arth
from analysis_utils import subjects, feature_renaming


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
if __name__ == '__main__':
    mode = args.mode
    datapath = args.datapath
    output_fname = args.output_fname
    features = arth.get_feat_cols(mode)
    
    df = pd.read_csv(datapath)
    df = df.loc[df['PatientID'].isin(subjects)]
    df = arth.core.remove_dead_electrodes(df, th_col='Thresholds')

    X = df[features].to_numpy()
    y = df['Thresholds'].to_numpy()

    correlation_results = []
    for i, feature in enumerate(features):
        correlation, p = scipy.stats.spearmanr(X[:,i], y)
        correlation = np.abs(correlation)
        correlation_results.append((feature, correlation, p))
        
    correlation_results = sorted(correlation_results, key=lambda x: x[1])
    features_sorted = [result[0] for result in correlation_results]
    absolute_correlations_sorted = [result[1] for result in correlation_results]
    p_values_sorted = [result[2] for result in correlation_results]

    fig, ax = plt.subplots(figsize=(8, 15))
    ax.barh(np.arange(X.shape[-1]), absolute_correlations_sorted, color='steelblue', edgecolor='k', alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 28)
    ax.set_yticks(np.arange(X.shape[-1]))
    ax.set_yticklabels([feature_renaming[feat_name] for feat_name in features_sorted])
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_xlabel('Absolute Spearman Correlation Coefficient', size=label_fontsize)
    ax.grid(alpha=0.2)

    for i in range(len(p_values_sorted)):
        p = p_values_sorted[i]
        if p > 0.05:
            ax.annotate('n.s.', (absolute_correlations_sorted[i]+0.06, i-0.1), ha='center', va='center', size=tick_fontsize)

    plt.savefig(output_fname, transparent=True, bbox_inches='tight')