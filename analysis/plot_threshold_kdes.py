import sys
import numpy as np
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('..')
import argus_thresholds as arth
from analysis_utils import subjects

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', \
                    type=str, \
                    required=True, \
                    help='Path to csv file containing processed data of all subjects')
parser.add_argument('--output_fname', \
                    type=str, \
                    required=True, \
                    help='Output filename for saving figure')
args = parser.parse_args()

num_rows = 3
num_cols = 4
text_size = 12
if __name__ == '__main__':
    datapath = args.datapath
    output_fname = args.output_fname
    
    df = pd.read_csv(datapath)
    df = df.loc[df['PatientID'].isin(subjects)]
    df = arth.core.remove_dead_electrodes(df, th_col='Thresholds')
    
    subject_thresholds = {}
    for subject in subjects:
        subject_df = df[df['PatientID'] == subject].copy()
        subject_thresholds[subject] = subject_df['Thresholds'].to_numpy()

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 3*num_rows), sharex=False, sharey=True)
    for ax, subject in zip(axes.ravel(), subjects):
        thresholds = subject_thresholds[subject]
        sns.kdeplot(thresholds, ax=ax, clip=[0, 999], fill=True, color='steelblue')
        ax.set_xlim(0, 999)
        ax.set_ylabel('')
        ax.set_title('Subject {}'.format(subject))
        ax.grid(alpha=0.6)

    fig.text(-0.01, 0.5, 'Density', ha='center', va='center', rotation=90, size=text_size)
    fig.text(0.5, -0.01, 'Perceptual Threshold (µA)', ha='center', va='center', rotation=0, size=text_size)
    plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    plt.savefig(output_fname, transparent=True, bbox_inches='tight')