import numpy as np
import pandas as pd

# Ignore all warnings/status messages from pulse2percept:
import logging
logging.getLogger("pulse2percept").setLevel(logging.ERROR)
from pulse2percept.implants import ArgusII
try:
    # p2p <= 0.5
    from pulse2percept.retina import ret2dva
except ImportError:
    # p2p >= 0.6
    from pulse2percept.utils import Watson2014Transform
    ret2dva = Watson2014Transform.ret2dva

from scipy.interpolate import interp1d

from sklearn.preprocessing import scale
from sklearn.manifold import TSNE, MDS, SpectralEmbedding

from .utils import days_btw_cols


__all__ = ['add_feat_fundus', 'add_feat_neighbors', 'add_feat_mean_impedance',
           'add_feat_dead_electrodes', 'add_feat_charge_density',
           'add_feat_imp2th', 'add_feat_imp2height', 'add_feat_imp2height2th',
           'add_feat_first_visit', 'add_feat_last_visit', 'embed_tsne', 'embed_mds', 
           'embed_spectral', 'scale_electrode_thresholds', 
           'add_feat_time_since_first_measurement', 
           'add_feat_time_since_last_electrode_measurement', 
           'add_feat_last_thresholds', 'add_feat_last_impedance']


# Ganglion cell density as a function of eccentricity (Watson, 2014)

# Temporal:
# Eccentricity in degrees of visual angle (dva):
deg_temp = [0.183, 0.36, 0.73, 1.087, 1.871, 3.3, 3.61, 5.391,
            7.235, 10.7, 18.144, 26.144, 30.01, 33.042, 40.784,
            60.99, 76.241, 88.182]
# Ganglion cell density in number of cells per degree squared:
dense_temp = [44.424, 90.603, 341.455, 1011.025, 1827.699, 2326.305,
              2377.883, 2084.719, 1286.838, 490.316, 245.74, 206.199,
              178.806, 145.179, 91.602, 47.445, 22.758, 10.68]
rgc_temp = interp1d(deg_temp, dense_temp)

# Nasal:
deg_nas = [0.185, 0.378, 0.553, 0.722, 1.094, 1.838, 3.381, 5.491,
           7.411, 11.631, 19.635, 23.373, 37.613, 48.421, 59.794, 64.645]
dense_nas = [13.594, 51.23, 117.877, 296.093, 820.891, 1584.893,
             2039.5, 1711.328, 1231.618, 547.136, 153.361, 110.372,
             31.278, 18.277, 9.676, 8.767]
rgc_nas = interp1d(deg_nas, dense_nas)

# Superior:
deg_sup = [0.545,  1.075,  1.414,  1.816,  3.262,  3.568,  5.461, 10.897,
           22.958, 26.194, 33.516, 40.898, 45.297, 53.58, 62.273, 71.566,
           77.843, 84.714]
dense_sup = [28.651,  688.803, 1344.533, 1847.85, 2107.704, 1995.262,
             1374.343,  429.866,  109.168,   97.831,   59.73,   31.278,
             25.959,   21.544,   15.676,    8.672,    7.602,    5.471]
rgc_sup = interp1d(deg_sup, dense_sup)

# Inferior:
deg_inf = [0.183, 0.364, 0.73, 1.061, 1.447, 1.783, 3.223, 5.429,
           7.422, 9.214, 14.737, 22.848, 30.306, 37.633, 45.338, 54.264,
           58.676, 64.24, 74.843]
dense_inf = [37.276, 90.603, 506.713, 1068., 1637.894, 1951.984,
             2130.941, 1359.356, 696.397, 484.969, 143.596, 69.64,
             40.693, 25.396, 18.277, 16.379, 13.895, 9.891, 2.834]
rgc_inf = interp1d(deg_inf, dense_inf)


def get_rgc_dense(rho, theta):
    theta = (np.rad2deg(theta) + 360) % 360
    try:
        if theta >= 0 and theta < 90:
            # between temporal and superior
            return (90 - theta) / 90 * rgc_temp(rho) + (theta - 0) / 90 * rgc_sup(rho)
        elif theta >= 90 and theta < 180:
            # between superior and nasal
            return (180 - theta) / 90 * rgc_sup(rho) + (theta - 90) / 90 * rgc_nas(rho)
        elif theta >= 180 and theta < 270:
            # between nasal and inferior
            return (270 - theta) / 90 * rgc_nas(rho) + (theta - 180) / 90 * rgc_inf(rho)
        else:
            # between inferior and temporal
            return (360 - theta) / 90 * rgc_inf(rho) + (theta - 270) / 90 * rgc_temp(rho)
    except ValueError:
        return 0


def add_feat_fundus(dff):
    df = dff.copy()

    # Set new feature columns to default value:
    df['ElectrodeLocRho (µm)'] = np.nan
    df['ElectrodeLocTheta (rad)'] = np.nan
    df['ImplantMeanLocRho (µm)'] = np.nan
    df['ImplantMeanLocTheta (rad)'] = np.nan
    df['ImplantMeanRot (rad)'] = np.nan
    df['OpticDiscElectrodeDist (µm)'] = np.nan
    df['OpticDiscLocX (µm)'] = np.nan
    df['OpticDiscLocY (µm)'] = np.nan

    # Extract location of implant and optic disc from combined data frame:
    cols = ['PatientID', 'implant_x', 'implant_y', 'implant_rot',
            'loc_od_x', 'loc_od_y']
    mat = df.loc[:, cols].drop_duplicates().dropna().set_index('PatientID')

    # For each subject, need to generate the implant, then look at the location
    # of each electrode and merge that into df:
    for subject_id, row in mat.iterrows():
        # Create the implant so we can access the location of each electrode:
        try:
            implant = ArgusII(x_center=row['implant_x'],
                              y_center=row['implant_y'],
                              rot=row['implant_rot'])
        except TypeError:
            # p2p >= 0.6:
            implant = ArgusII(x=row['implant_x'], y=row['implant_y'],
                              rot=row['implant_rot'])

        # Add implant location:
        implant_rho = ret2dva(np.sqrt(row['implant_x'] ** 2
                                      + row['implant_y'] ** 2))
        implant_th = np.arctan2(row['implant_y'], row['implant_x'])
        slc = df.PatientID == subject_id
        df.loc[slc, 'ImplantMeanLocRho (µm)'] = implant_rho
        df.loc[slc, 'ImplantMeanLocTheta (rad)'] = implant_th
        df.loc[slc, 'ImplantMeanRot (rad)'] = row['implant_rot']
        df.loc[slc, 'OpticDiscLocX (µm)'] = row['loc_od_x']
        df.loc[slc, 'OpticDiscLocY (µm)'] = row['loc_od_y']

        # Find all electrodes for this subject:
        for ename in df[df.PatientID == subject_id].ElectrodeLabel.unique():
            # Find the electrode in the implant:
            electrode = implant['%s%d' % (ename[0], int(ename[1:]))]
            if not electrode:
                raise ValueError("Could not find Electrode %s for Subject "
                                 "%s" % (ename, subject_id))

            # Find entries in df corresponding to this subject and electrode:
            slc = (df.PatientID == subject_id) & (df.ElectrodeLabel == ename)

            # Store electrode location in polar coordinates:
            try:
                df.loc[slc, 'ElectrodeLocRho (µm)'] = np.sqrt(
                    electrode.x_center ** 2 + electrode.y_center ** 2
                )
                df.loc[slc, 'ElectrodeLocTheta (rad)'] = np.arctan2(
                    electrode.y_center, electrode.x_center
                )

                # Dist to OD
                dist_od = np.sqrt((electrode.x_center - row['loc_od_x']) ** 2
                                  + (electrode.y_center - row['loc_od_y']) ** 2)
                df.loc[slc, 'OpticDiscElectrodeDist (µm)'] = dist_od

                # Polar coordinates:
                rho = ret2dva(np.sqrt(electrode.x_center ** 2
                                      + electrode.y_center ** 2))
                theta = np.arctan2(electrode.y_center, electrode.x_center)
            except AttributeError:
                # p2p >= 0.6:
                df.loc[slc, 'ElectrodeLocRho (µm)'] = np.sqrt(
                    electrode.x ** 2 + electrode.y ** 2
                )
                df.loc[slc, 'ElectrodeLocTheta (rad)'] = np.arctan2(
                    electrode.y, electrode.x
                )

                # Dist to OD
                dist_od = np.sqrt((electrode.x - row['loc_od_x']) ** 2
                                  + (electrode.y - row['loc_od_y']) ** 2)
                df.loc[slc, 'OpticDiscElectrodeDist (µm)'] = dist_od

                # Polar coordinates:
                rho = ret2dva(np.sqrt(electrode.x ** 2
                                      + electrode.y ** 2))
                theta = np.arctan2(electrode.y, electrode.x)

            df.loc[slc, 'ElectrodeLocRho (µm)'] = rho
            df.loc[slc, 'ElectrodeLocTheta (rad)'] = theta
            df.loc[slc, 'RGCDensity (cells/deg2)'] = get_rgc_dense(rho, theta)
    dropcols = ['implant_x', 'implant_y',
                'implant_rot', 'loc_od_x', 'loc_od_y']
    return df.drop(columns=dropcols)


def add_feat_mean_impedance(dff):
    """Mean impedance/std impedance for a given subject/day"""
    df = dff.copy()
    df['ImpedanceCV (std/mu)'] = 0
    for (subj, date), _ in df.groupby(['PatientID', 'TestDate (timestamp)']):
        idx = ((df['PatientID'] == subj) & (df['TestDate (timestamp)'] == date)
               & (df['Impedance (kΩ)'] > 0))
        imp = df.loc[idx, 'Impedance (kΩ)'].dropna()
        if len(imp) > 1:
            df.loc[idx, 'ImpedanceCV (std/mu)'] = imp.std() / imp.mean()

    return df


def add_feat_dead_electrodes(dff):
    """Fraction of electrodes for a given subjet/day with thresh == 999uA"""
    df = dff.copy()
    # Engineer features for dead electrodes:
    df['ElectrodesDead (frac)'] = 0.0
    for (subj, date), dd in df.groupby(['PatientID', 'TestDate (timestamp)']):
        # Electrodes aren't just "dead": on some dates, they give a measurable
        # threshold, on others they don't.
        n_measured = len(dd.ElectrodeLabel.unique())
        n_dead = len(dd[dd['Thresholds (µA)'] == 999].ElectrodeLabel.unique())
        idx = (df['PatientID'] == subj) & (df['TestDate (timestamp)'] == date)
        df.loc[idx, 'ElectrodesDead (frac)'] = n_dead / n_measured
    return df


def add_feat_charge_density(dff, cd_thresh):
    df = dff.copy()
    # Engineer features for how many electrodes <= cd_thresh:
    label = 'ElectrodesCD%.2fmC/cm2' % cd_thresh
    df[label + ' (frac)'] = np.nan
    for (subj, date), dd in df.groupby(['PatientID', 'TestDate (timestamp)']):
        n_measured = len(dd.ElectrodeLabel.unique())
        slc = dd[dd['ChargeDensityLimit (mC/cm2)'] <= cd_thresh]
        n_lower = len(slc.ElectrodeLabel.unique())
        idx = (df['PatientID'] == subj) & (df['TestDate (timestamp)'] == date)
        df.loc[idx, label + ' (frac)'] = n_lower / n_measured
    return df


def add_feat_neighbors(dff, col, k=1, std=True, default='nan', verbose=False):
    """Add mean threshold of k-nearest neighbors

    Parameters
    ----------
    k : int
        k-nearest neighbor. For example, k=1 neighbors for C3 are B2-B4, C2,
        C4, D2-D4. k=2 neighbors for C3 are A1-A5, B1, B5, C1, C5, D1, D5,
        E1-E5. Note that k=2 neighbors do not include k=1 neighbors.
    default: {'nan', 'mean'}
    """
    el_dist = 525.0  # Argus II
    feat_name_mu = '%sMuK%d (µA)' % (col.split(' ')[0], k)
    feat_name_std = '%sStdK%d (µA)' % (col.split(' ')[0], k)

    # Generate names of neighbors for each electrode:
    # fix bugs in finding neighbours
    neighbors = {}
    k = 1
    for erow in range(ord('A'), ord('F') + 1):
        for ecol in range(1, 11):
            ename = '{}{:02}'.format(chr(erow), ecol)
            nghb = []
            for i in range(1, k + 1):
                if erow - i >= ord('A'):
                    nghb += ['{}{:02}'.format(chr(erow - i), c)
                             for c in range(max(1, ecol - k), min(10, ecol + k) + 1)]
                if erow + i <= ord('F'):
                    nghb += ['{}{:02}'.format(chr(erow + i), c)
                             for c in range(max(1, ecol - k), min(10, ecol + k) + 1)]
            nghb += ['{}{:02}'.format(chr(erow), c)
                     for c in range(max(1, ecol - k), ecol)]
            nghb += ['{}{:02}'.format(chr(erow), c)
                     for c in range(ecol + 1, min(10, ecol + k) + 1)]
            neighbors[ename] = set(nghb)
    if verbose:
        print(neighbors)

    # Calculate mean threshold of neighbors:
    df = dff.copy()
    df[feat_name_mu] = np.nan
    if std:
        df[feat_name_std] = np.nan
    for (subj, date), slc in df.groupby(['PatientID', 'TestDate (timestamp)']):
        # Default 'mean': if no neighbors given, assign the mean across the
        # array
        if default == 'mean':
            df.loc[slc.index, feat_name_mu] = np.mean(slc[col])
            if std:
                df.loc[slc.index, feat_name_std] = np.std(slc[col])
        # For each electrode, check if neighbors are available:
        for idx0, row0 in slc.iterrows():
            # get electrode lable
            e0 = row0['ElectrodeLabel']
            # Check if any neighbors are available:
            th_ngh = []
            for neighbor in neighbors[e0]:
                row1 = slc[slc['ElectrodeLabel'] == neighbor]
                if len(row1) > 0:
                    th_ngh.append(row1[col].values[0])
            # If any neighbors found, add their mean threshold as feature:
            if len(th_ngh) > 0:
                df.loc[idx0, feat_name_mu] = np.mean(th_ngh)
                if std:
                    df.loc[idx0, feat_name_std] = np.std(th_ngh)
                if verbose:
                    print(subj, date, row0['ElectrodeLabel'], th_ngh)
    return df


raw_imp2th = np.array([1.0, 1698.6,
                       1.2, 1330.1,
                       1.5, 1161.2,
                       1.8, 897.0,
                       2.3, 721.8,
                       2.8, 580.8,
                       3.4, 480.2,
                       4.2, 391.7,
                       5.2, 319.5,
                       6.1, 267.7,
                       7.5, 218.4,
                       8.8, 188.1,
                       10.1, 164.2,
                       11.4, 143.3,
                       13.1, 126.8,
                       15.4, 106.3,
                       18.2, 89.1,
                       20.9, 78.8,
                       23.5, 70.7,
                       26.9, 61.7,
                       30.1, 53.9,
                       33.3, 49.0,
                       36.6, 45.8,
                       40.5, 41.1,
                       44.5, 36.8,
                       49.3, 34.0,
                       53.0, 31.3,
                       57.9, 29.2,
                       61.8, 27.0,
                       67.0, 25.2]).reshape((-1, 2))

imp2th = interp1d(raw_imp2th[:, 0], raw_imp2th[:, 1], kind='linear',
                  bounds_error=False, fill_value='extrapolate')


def add_feat_imp2th(dff):
    """Predict threshold from impedance (deBalthasar et al., 2008)"""
    df = dff.copy()
    df['Impedances2Thresholds (µA)'] = np.nan
    for (subj, date), _ in df.groupby(['PatientID', 'TestDate (timestamp)']):
        idx = ((df['PatientID'] == subj) & (df['TestDate (timestamp)'] == date)
               & (df['Impedance (kΩ)'] > 0))
        df.loc[idx, 'Impedances2Thresholds (µA)'] = imp2th(
            df.loc[idx, 'Impedance (kΩ)']
        )
    return df


raw_height2imp = np.array([54.4, 55.4,
                           65.4, 48.4,
                           78.6, 41.5,
                           95.7, 34.8,
                           111.7, 31.0,
                           129.8, 27.6,
                           147.8, 24.4,
                           172.5, 21.5,
                           200.4, 19.1,
                           235.0, 16.9,
                           262.6, 15.3,
                           289.2, 14.3,
                           315.5, 13.4,
                           349.1, 12.3,
                           386.4, 11.1,
                           438.0, 10.0,
                           482.4, 9.2,
                           560.3, 8.2,
                           614.1, 7.6,
                           647.6, 7.3,
                           692.9, 6.9,
                           744.9, 6.5,
                           793.2, 6.2,
                           840.5, 5.9,
                           886.3, 5.7,
                           943.7, 5.4,
                           1000.0, 5.1]).reshape((-1, 2))
imp2height = interp1d(raw_height2imp[:, 1], raw_height2imp[:, 0],
                      kind='linear', bounds_error=False,
                      fill_value='extrapolate')


def add_feat_imp2height(dff):
    """Predict electrode height from impedance (deBalthasar et al., 2008)"""
    df = dff.copy()
    df['Impedances2Height (µm)'] = np.nan
    for (subj, date), _ in df.groupby(['PatientID', 'TestDate (timestamp)']):
        idx = ((df['PatientID'] == subj) & (df['TestDate (timestamp)'] == date)
               & (df['Impedance (kΩ)'] > 0))
        df.loc[idx, 'Impedances2Height (µm)'] = imp2height(
            df.loc[idx, 'Impedance (kΩ)']
        )
    return df


raw_height2th = np.array([54.5, 23.2,
                          63.0, 26.5,
                          75.2, 29.4,
                          88.0, 33.5,
                          106.0, 38.9,
                          126.5, 45.7,
                          146.4, 53.0,
                          166.9, 61.4,
                          194.2, 72.3,
                          219.2, 82.5,
                          253.7, 100.0,
                          283.4, 117.6,
                          328.1, 142.5,
                          368.4, 172.7,
                          420.0, 215.4,
                          464.5, 257.2,
                          493.4, 289.4,
                          529.5, 325.7,
                          565.4, 366.5,
                          603.7, 412.5,
                          638.1, 450.7,
                          681.3, 485.2,
                          709.3, 522.3,
                          746.0, 562.3,
                          780.6, 596.5,
                          816.9, 651.8,
                          850.5, 701.7,
                          876.6, 744.4,
                          903.5, 778.1,
                          921.9, 813.3,
                          945.4, 850.1,
                          979.4, 888.6
                          ]).reshape((-1, 2))
height2th = interp1d(raw_height2th[:, 0], raw_height2th[:, 1],
                     kind='linear', bounds_error=False, fill_value='extrapolate')


def add_feat_imp2height2th(dff):
    """Predict electrode height from impedance (deBalthasar et al., 2008)"""
    df = dff.copy()
    df['Impedances2Heights2Thresholds (µA)'] = np.nan
    for (subj, date), _ in df.groupby(['PatientID', 'TestDate (timestamp)']):
        idx = ((df['PatientID'] == subj) & (df['TestDate (timestamp)'] == date)
               & (df['Impedance (kΩ)'] > 0))
        pred_th = height2th(imp2height(df.loc[idx, 'Impedance (kΩ)']))
        df.loc[idx, 'Impedances2Heights2Thresholds (µA)'] = pred_th
    return df


def add_feat_first_visit(dff, remove_first_visit_samples=False):
    df = dff.copy()

    # Need the fraction of dead electrodes for the first visit. Easiest way
    # right now is to do it for all, but we could re-write this to only
    # calculate it for `idx`:
    df = add_feat_dead_electrodes(df)

    groupby = df.groupby(['PatientID', 'ElectrodeLabel'])
    # Get the index of the first measured threshold:
    idx = groupby['TestDate (timestamp)'].apply(lambda x: x.idxmin()).values
    tmp_df = df.loc[idx][['PatientID', 'ElectrodeLabel',
                          'TestDate (YYYYmmdd)', 'Thresholds (µA)',
                          'Impedance (kΩ)', 'ElectrodesDead (frac)',
                          'FalsePositiveRate', 'ChargeDensityLimit (mC/cm2)',
                          'MaxCurrent (µA)']]

    # Rename columns to avoid ambiguity:
    tmp_df.rename(
        columns={
            'TestDate (YYYYmmdd)': 'FirstTestDate (YYYYmmdd)',
            'Thresholds (µA)': 'FirstThresholds (µA)',
            'Impedance (kΩ)': 'FirstImpedance (kΩ)',
            'FalsePositiveRate': 'FirstFalsePositiveRate',
            'ElectrodesDead (frac)': 'FirstElectrodesDead (frac)',
            'ChargeDensityLimit (mC/cm2)': 'FirstChargeDensityLimit (mC/cm2)',
            'MaxCurrent (µA)': 'FirstMaxCurrent (µA)'
        },
        inplace=True)

    # Merge the new features:
    df = df.merge(tmp_df, left_on=['PatientID', 'ElectrodeLabel'],
                  right_on=['PatientID', 'ElectrodeLabel'])

    # Store the time since first visit:
    diff = days_btw_cols(df['TestDate (YYYYmmdd)'],
                         df['FirstTestDate (YYYYmmdd)'],
                         fmt_now="%Y%m%d", fmt_then="%Y%m%d")
    df['TimeSinceFirst (days)'] = pd.Series(diff, index=df.index)
    df.drop(columns=['FirstTestDate (YYYYmmdd)', 'ElectrodesDead (frac)'],
            inplace=True)

    # Remove the first visit from the dataset:
    if remove_first_visit_samples:
        df.drop(index=idx, inplace=True)

    return df


def add_feat_last_visit(dff):
    df = dff.copy()

    # Need the fraction of dead electrodes for the first visit. Easiest way
    # right now is to do it for all, but we could re-write this to only
    # calculate it for `idx`:

    target_cols = ['TestDate (YYYYmmdd)', 'Thresholds (µA)',
                   'Impedance (kΩ)', 'ElectrodesDead (frac)',
                   'FalsePositiveRate', 'ChargeDensityLimit (mC/cm2)',
                   'MaxCurrent (µA)']

    history_cols = ['LastTestDate (YYYYmmdd)', 'LastThreshold (µA)',
                    'LastImpedance (kΩ)',     'LastElectrodesDead (frac)',
                    'LastFalsePositiveRate',  'LastChargeDensityLimit (mC/cm2)',
                    'LastMaxCurrent (µA)']

    groupby = df.sort_values(by=['TestDate (timestamp)']).groupby(
        ['PatientID', 'ElectrodeLabel'])

    df[history_cols] = groupby[target_cols].diff()
    
    df['LastTestDate (YYYYmmdd)'] = np.array(df['LastTestDate (YYYYmmdd)'], dtype = np.int64)
    for t_col, h_col in zip(target_cols, history_cols):
        df[h_col] = df[t_col].subtract(df[h_col])

    # Store the time since last visit:
    diff = days_btw_cols(df['TestDate (YYYYmmdd)'],
                         df['LastTestDate (YYYYmmdd)'],
                         fmt_now="%Y%m%d", fmt_then="%Y%m%d")
    df['TimeSinceLast (days)'] = pd.Series(diff, index=df.index)

    return df


def add_feat_time_since_first_measurement(dff):
    """
    Add the amount of time since the first meaurement for
    the patient (in days).

    Parameters
    ----------
    dff: pd.DataFrame
        Base feature set to which we will add new feature to
        
    Returns
    ----------
    df: pd.DataFrame
        Feature set containing with new feature \'TimeSinceFirstMeasurement\'
    """
    assert 'SubjectTimePostOp (days)' in dff.columns, \
           'Need feature \"SubjectTimePostOp (days)\" for computation of'\
           'TimeSinceFirstMeasurement'
        
    df = dff.copy()
    df['TimeSinceFirstMeasurement (days)'] = df['SubjectTimePostOp (days)'].copy()
    for patient_id in df['PatientID'].unique():
        patient_idx = df[df['PatientID'] == patient_id].index
        first_testdate = df.loc[patient_idx, 'SubjectTimePostOp (days)'].min()
        df.loc[patient_idx, 'TimeSinceFirstMeasurement (days)'] = \
            df.loc[patient_idx, 'TimeSinceFirstMeasurement (days)'].apply(lambda x: x-first_testdate)
    return df


def add_feat_time_since_last_electrode_measurement(dff):
    """
    Add the amount of time since the last measurement for each given
    electrode (days).

    Parameters
    ----------
    dff: pd.DataFrame
        Base feature set to which we will add new feature to
        
    Returns
    ----------
    df: pd.DataFrame
        Feature set containing with new feature \'TimeSinceLastElectrodeMeasurement\'
    """
    def get_last_measurement_day(measurement_days, day):
        """
        Get most recent, historical electrode measurement day
        
        Parameters
        ----------
        measurement_days: np.array or list
            List of days for which threshold was measured for the electrode
        day: int
            Current measurement day
            
        Returns
        ----------
        last_measurement_day: int
            Most recent, historical electrode measurement day
        
        """
        measurement_days = np.array(measurement_days)
        previous_day_inds = np.where(measurement_days < day)[0]
        if len(previous_day_inds) == 0:
            last_measurement_day = measurement_days[0]
        else:
            last_measurement_day = measurement_days[previous_day_inds[-1]]
        return last_measurement_day
    
    assert 'SubjectTimePostOp (days)' in dff.columns, \
           'Need feature \"SubjectTimePostOp (days)\" for computation of'\
           'TimeSinceLastElectrodeMeasurement'
        
    df = dff.copy()
    df['TimeSinceLastElectrodeMeasurement (days)'] = df['SubjectTimePostOp (days)'].copy()
    for patient_id in df['PatientID'].unique():
        patient_df = df[df['PatientID'] == patient_id]
        for electrode_label in patient_df['ElectrodeLabel'].unique():
            patient_electrode_idx = patient_df[patient_df['ElectrodeLabel'] == electrode_label].index
            electrode_measurement_days = df.loc[patient_electrode_idx,'SubjectTimePostOp (days)'].unique()
            df.loc[patient_electrode_idx,'TimeSinceLastElectrodeMeasurement (days)'] = \
                df.loc[patient_electrode_idx,'TimeSinceLastElectrodeMeasurement (days)']\
                .apply(lambda x: x - get_last_measurement_day(electrode_measurement_days, x))
    return df


def add_feat_last_thresholds(dff):
    """
    Add the the previous threshold measurement for each electrode.

    Parameters
    ----------
    dff: pd.DataFrame
        Base feature set to which we will add new feature to
        
    Returns
    ----------
    df: pd.DataFrame
        Feature set containing with new feature \'LastThresholds\'
    """
    assert 'SubjectTimePostOp (days)' in dff.columns, \
           'Need feature \"SubjectTimePostOp (days)\" for computation of'\
           'LastThresholds'
        
    df = dff.copy()
    df.sort_values('SubjectTimePostOp (days)', inplace=True)
    df['LastThresholds (µA)'] = np.nan
    for patient_id in df['PatientID'].unique():
        patient_df = df[df['PatientID'] == patient_id]
        for electrode_label in patient_df['ElectrodeLabel'].unique():
            patient_electrode_idx = patient_df[patient_df['ElectrodeLabel'] == electrode_label].index
            for i in range(1, len(patient_electrode_idx)):
                for j in range(1, i+1):
                    # Make sure previous measurement is coming from a different time
                    prev_timestamp = df.loc[patient_electrode_idx[i-j], 'TestDate (timestamp)']
                    curr_timestamp = df.loc[patient_electrode_idx[i], 'TestDate (timestamp)']
                    if prev_timestamp == curr_timestamp:
                        continue
                    else:
                        previous_threshold = df.loc[patient_electrode_idx[i-j], 'Thresholds (µA)']
                        df.loc[patient_electrode_idx[i], 'LastThresholds (µA)'] = previous_threshold
                        break
                        
    return df


def add_feat_last_impedance(dff):
    """
    Add the the previous impedance measurement for each electrode (kΩ).

    Parameters
    ----------
    dff: pd.DataFrame
        Base feature set to which we will add new feature to
        
    Returns
    ----------
    df: pd.DataFrame
        Feature set containing with new feature \'LastImpedance\'
    """
    assert 'SubjectTimePostOp (days)' in dff.columns, \
           'Need feature \"SubjectTimePostOp (days)\" for computation of'\
           'LastImpedance'
        
    df = dff.copy()
    df.sort_values('SubjectTimePostOp (days)', inplace=True)
    df['LastImpedance (kΩ)'] = np.nan
    for patient_id in df['PatientID'].unique():
        patient_df = df[df['PatientID'] == patient_id]
        for electrode_label in patient_df['ElectrodeLabel'].unique():
            patient_electrode_idx = patient_df[patient_df['ElectrodeLabel'] == electrode_label].index
            for i in range(1, len(patient_electrode_idx)):
                for j in range(1, i+1):
                    # Make sure previous measurement is coming from a different time
                    prev_timestamp = df.loc[patient_electrode_idx[i-j], 'TestDate (timestamp)']
                    curr_timestamp = df.loc[patient_electrode_idx[i], 'TestDate (timestamp)']
                    if prev_timestamp == curr_timestamp:
                        continue
                    else:
                        previous_impedance = df.loc[patient_electrode_idx[i-j], 'Impedance (kΩ)']
                        df.loc[patient_electrode_idx[i], 'LastImpedance (kΩ)'] = previous_impedance
                        break
    return df


def scale_electrode_thresholds(dff, scaling_type='first'):
    """
    Scale electrode threshold measurements.

    Parameters
    ----------
    dff: pd.DataFrame
        Base feature set with threshold measurements in column
        Thresholds (µA).
        
    scaling_type: string
        Method of threshold scaling.  Implemented methods include:
            - 'first': For each patient, divide each threshold measurement
                       by the first measured threshold of that electrode.
        
    Returns
    ----------
    df: pd.DataFrame
        Feature set with scaled threshold measurements in column
        Thresholds (scaled).
    """
    df = dff.copy()
    if np.any((df['Thresholds (µA)'] == 999) | (df['Thresholds (µA)'] == 0)):
        warnings.warn('Electrodes with Thresholds (µA) in {0,999} exist,' \
                      'scaling may result in unexpected threshold values')
    if scaling_type not in ['first']:
        warnings.warn('Scaling type %s not yet implemented. Skipping.' % scaling_type)
        
    df['Thresholds (scaled)'] = df['Thresholds (µA)'].copy()
    if scaling_type == 'first':
        for patient_id in df['PatientID'].unique():
            patient_df = df[df['PatientID'] == patient_id]
            for electrode_label in patient_df['ElectrodeLabel'].unique():
                patient_electrode_idx = \
                    patient_df[patient_df['ElectrodeLabel'] == electrode_label].index
                first_electrode_measurement_idx = \
                    df.loc[patient_electrode_idx,'TestDate (timestamp)'].idxmin()
                first_electrode_threshold = \
                    df.loc[first_electrode_measurement_idx, 'Thresholds (µA)']
                df.loc[patient_electrode_idx, 'Thresholds (scaled)'] = \
                    df.loc[patient_electrode_idx, 'Thresholds (scaled)']\
                    .apply(lambda x: x/first_electrode_threshold)
                df.loc[patient_electrode_idx, 'FirstActiveThresholds (µA)'] = \
                    first_electrode_threshold
    return df


def embed_tsne(dff, n_components=3, n_jobs=-1):
    """Meta feature: t-SNE embedding"""
    df = dff.copy()
    drop_cols = []
    for col in ['PatientID', 'ElectrodeLabel', 'Thresholds (µA)']:
        if col in df.columns:
            drop_cols.append(col)
    # Calculate t-SNE embedding:
    tsne = TSNE(n_components=n_components, n_jobs=n_jobs)
    df_scaled = scale(df.drop(columns=drop_cols).fillna(0))
    return tsne.fit_transform(df_scaled)


def embed_mds(dff, n_components=2, n_jobs=-1):
    """Meta feature: Multi-dimensional scaling"""
    df = dff.copy()
    drop_cols = []
    for col in ['PatientID', 'ElectrodeLabel', 'Thresholds (µA)']:
        if col in df.columns:
            drop_cols.append(col)
    # Calculate MSE embedding:
    mds = MDS(n_components=n_components, n_jobs=n_jobs)
    df_scaled = scale(df.drop(columns=drop_cols).fillna(0))
    return mds.fit_transform(df_scaled)


def embed_spectral(dff, n_components=2, n_jobs=-1):
    """Meta feature: Spectral embedding"""
    df = dff.copy()
    drop_cols = []
    for col in ['PatientID', 'ElectrodeLabel', 'Thresholds (µA)']:
        if col in df.columns:
            drop_cols.append(col)
    # Calculate MSE embedding:
    spec = SpectralEmbedding(n_components=n_components, n_jobs=n_jobs)
    df_scaled = scale(df.drop(columns=drop_cols).fillna(0))
    return spec.fit_transform(df_scaled)
    for c in range(df_embedded.shape[1]):
        col = 'Spectral_%d' % c
        df[col] = df_embedded[:, c]
    return df
