#!/usr/bin/env python
# coding: utf-8
# Preprocesses and stores the data

import sys
sys.path.append('..')
import argparse
import numpy as np
from os import environ as ose
from os.path import join as opjoin
from sklearn.preprocessing import StandardScaler
import argus_thresholds as arth


parser = argparse.ArgumentParser()
# Feature set to extract ('routine', 'fitting', or 'followup'). Default: 'followup'
parser.add_argument('--mode', \
                    choices=['routine', 'fitting', 'followup'], \
                    type=str, \
                    required=True, \
                    help='Feature sets to compute')
# Standardize feature values
parser.add_argument('--standardize', \
                    default=False, \
                    action='store_true', \
                    help='Standardize feature values')
# Scale threshold measurements
parser.add_argument('--scale_thresholds', \
                    nargs='?', \
                    type=str, \
                    help='Scale threshold values')
# Ignore outlier measurements
parser.add_argument('--ignore_outliers', \
                    default=False, \
                    action='store_true', \
                    help='Standardize feature values')
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
# Calculate meta features (vector embeddings like MDS, by default off):
parser.add_argument('--calc_meta', \
                    default=False, \
                    action='store_true', \
                    help='calculate meta features (e.g., vector embeddings)')
args = parser.parse_args()


def main():
    mode = args.mode
    standardize = args.standardize
    scale_thresholds = args.scale_thresholds
    min_samples = args.min_samples
    ignore_outliers = args.ignore_outliers
    keep_nans = args.keep_nans
    calc_meta = args.calc_meta
    print('mode:', mode)
    print('standardize:', standardize)
    print('scale_thresholds:', scale_thresholds)
    print('ignore_outliers:', ignore_outliers)
    print('keep_nans:', keep_nans)
    print('min_samples:', min_samples)
    print('calc_meta:', calc_meta)
    
    # Set output filename
    str_standard = '-std' if standardize else ''
    str_meta = '-meta' if calc_meta else ''
    str_outliers = '-ignoreoutliers' if ignore_outliers else ''
    str_scale_thresholds = '-scalethresh%s' % scale_thresholds if scale_thresholds is not None else ''
    str_nans = '-keepnans' if keep_nans else ''
    fname = 'data-preprocessed-%s%s%s%s%s%s-2023.csv' % (
                mode, str_standard, str_scale_thresholds, str_outliers, str_nans, str_meta)

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

    # Calculate meta features (we have to be careful not to calculate e.g.
    # the MDS embedding of the tSNE features, so we do these step-by-step):
    if calc_meta:
        meta_features = [
            # (embedding_name, embedding_function, n_components)
            ('tSNE', arth.feat_eng.embed_tsne, 3),
            ('MDS', arth.feat_eng.embed_mds, 2),
            ('Spectral', arth.feat_eng.embed_spectral, 2)
        ]
        for embed_name, embed_fnc, n_components in meta_features:
            print('-', embed_name)
            # Embed only the `get_feat_cols` columns:
            df_embed = embed_fnc(Xy[arth.get_feat_cols(mode)],
                                 n_components=n_components)
            for c in range(df_embed.shape[1]):
                col = '%s_%d' % (embed_name, c)
                Xyy[col] = df_embed[:, c]

    if standardize:
        # Standardize all non-categorical features:
        cat_cols = ['ImplantEye_RE', 'ImplantSite_12',
                    'ImplantSite_14', 'ImplantSite_17', 'ImplantSite_51',
                    'ImplantSite_52', 'ImplantSite_61', 'ImplantSite_71']
        col_std = [c for c in arth.get_feat_cols(mode) if c not in cat_cols]
        Xyy[col_std] = StandardScaler().fit_transform(Xyy[col_std])
        
    if not keep_nans:
        Xyy.dropna(axis=0, inplace=True)
                
    # Remove subjects with insufficient data quantities
    Xyy = arth.filter_min_samples(Xyy, min_samples)
    print('Xyy shape:', Xyy.shape)

    # Save to file
    Xyy.to_csv(fname, index=False)

if __name__ == "__main__":
    main()
