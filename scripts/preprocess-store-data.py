#!/usr/bin/env python
# coding: utf-8
# Preprocesses and stores the data

import sys
sys.path.append('..')
import argparse
import numpy as np
from os import environ as ose
from os.path import join as opjoin
import argus_thresholds as arth


parser = argparse.ArgumentParser()
# Feature set to extract ('routine', 'fitting', or 'followup'). Default: 'followup'
parser.add_argument('--mode', \
                    choices=['routine', 'fitting', 'followup'], \
                    type=str, \
                    required=True, \
                    help='Feature sets to compute')
# Scale threshold measurements
parser.add_argument('--scale_thresholds', \
                    nargs='?', \
                    type=str, \
                    help='Scale threshold values')
# Ignore outlier measurements
parser.add_argument('--ignore_outliers', \
                    default=False, \
                    action='store_true', \
                    help='Drop samples with outlier threshold measurements')
# Keep rows with NaN values
parser.add_argument('--keep_nans', \
                    default=False, \
                    action='store_true', \
                    help='Keep rows with NaN feature values')
# Remove subjects with less than this number of data points:
parser.add_argument('--min_samples', \
                    nargs='?', \
                    default=50, \
                    type=int, \
                    help='Minimum number of samples per subject')
args = parser.parse_args()


def main():
    mode = args.mode
    scale_thresholds = args.scale_thresholds
    min_samples = args.min_samples
    ignore_outliers = args.ignore_outliers
    keep_nans = args.keep_nans
    print('mode:', mode)
    print('scale_thresholds:', scale_thresholds)
    print('ignore_outliers:', ignore_outliers)
    print('keep_nans:', keep_nans)
    print('min_samples:', min_samples)
    
    # Set output filename
    str_outliers = '-ignoreoutliers' if ignore_outliers else ''
    str_scale_thresholds = '-scalethresh%s' % scale_thresholds if scale_thresholds is not None else ''
    str_nans = '-keepnans' if keep_nans else ''
    fname = 'data-preprocessed-%s%s%s%s-2023.csv' % (
                mode, str_scale_thresholds, str_outliers, str_nans)

    # Read raw data from each subject
    Xyraw = arth.load_data(opjoin(ose['DATA_ROOT'], 'argus_thresholds'),
                           sfile='subjects-2020-corrected.csv')
    print('Xyraw:', Xyraw.shape)
        
    if ignore_outliers:
        Xy = arth.preprocess_data(Xyraw, ignore_dead_electrodes=True, \
                                  remove_outliers=True, threshold_scaling=scale_thresholds)
    else:
        Xy = arth.preprocess_data(Xyraw, ignore_dead_electrodes=False, \
                                  remove_outliers=False, threshold_scaling=scale_thresholds)
        
    # If mode is first visit data used as features, remove first visit data
    if mode in ['fitting', 'followup']:
        Xy = arth.remove_first_measured_row(Xy)

    # Choose feature columns:
    Xyy = Xy[arth.get_feat_cols(mode)].copy()

    # Need to keep the labels though:
    Xyy['PatientID'] = Xy['PatientID'].copy()
    Xyy['ElectrodeLabel'] = Xy['ElectrodeLabel'].copy()
    Xyy['Thresholds'] = Xy['Thresholds (µA)'].copy()
        
    if not keep_nans:
        Xyy.dropna(axis=0, inplace=True)
                
    # Remove subjects with insufficient data quantities
    Xyy = arth.filter_min_samples(Xyy, min_samples)
    print('Xyy shape:', Xyy.shape)

    # Save to file
    Xyy.to_csv(fname, index=False)

if __name__ == "__main__":
    main()
