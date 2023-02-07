import sys
import numpy as np
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('..')
from argus_thresholds import get_feat_cols
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

if __name__ == '__main__':
    datapath = args.datapath
    mode = args.mode
    output_fname = args.output_fname
    feature_names = get_feat_cols(mode)
    
    df = pd.read_csv(datapath)
    df = df.loc[df['PatientID'].isin(subjects)]
    df = df[feature_names]

    corr = df.corr()
    corr.columns = [feature_renaming[feat_name] for feat_name in corr.columns]
    corr.dropna(how='all', axis=0, inplace=True)
    corr.dropna(how='all', axis=1, inplace=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, mask=np.triu(np.ones(corr.shape), k=1).astype(np.bool),
                vmin=-1, vmax=1, cmap='coolwarm', linewidths=1,
                cbar_kws={"shrink": .75});
    ax.set_xticklabels(corr.columns, size=10)
    ax.set_yticklabels(corr.columns, size=10)
    plt.tight_layout()
    plt.savefig(output_fname, transparent=True)