# 12 analyzed subjects
subjects = ['12-001', '12-004', '12-005', '14-001', \
            '17-002', '51-001', '51-003', '51-009', \
            '52-001', '52-003', '61-004', '71-002']

# Feature name reformatting for plotting
feature_renaming = {'SubjectAge (days)': 'SubjectAge',
                    'SubjectTimeBlind (days)': 'SubjectTimeBlind',
                    'SubjectAgeAtDiagnosis (years)': 'SubjectAgeAtDiagnosis',
                    'SubjectAgeAtSurgery (years)': 'SubjectAgeAtSurgery',
                    'SubjectTimePostOp (days)': 'ImplantTime',
                    'Impedance (kΩ)': 'Impedance',
                    'ImpedanceCV (std/mu)': 'ImpedanceCV',
                    'ElectrodeLocRho (µm)': 'ElectrodeLocRho',
                    'ElectrodeLocTheta (rad)': 'ElectrodeLocTheta',
                    'ImplantMeanLocRho (µm)': 'ImplantMeanLocRho',
                    'ImplantMeanLocTheta (rad)': 'ImplantMeanLocTheta',
                    'ImplantMeanRot (rad)': 'ImplantMeanRot',
                    'OpticDiscLocX (µm)': 'OpticDiscLocX',
                    'OpticDiscLocY (µm)': 'OpticDiscLocY',
                    'RGCDensity (cells/deg2)': 'RGCDensity',
                    'Impedances2Thresholds (µA)': 'Impedances2Thresholds',
                    'Impedances2Height (µm)': 'Impedances2Height',
                    'Impedances2Heights2Thresholds (µA)': 'Impedances2Heights2Thresholds',
                    'FirstImpedance (kΩ)': 'FirstImpedance',
                    'FirstThresholds (µA)': 'FirstThresholds',
                    'FirstMaxCurrent (µA)': 'FirstMaxCurrent',
                    'FirstChargeDensityLimit (mC/cm2)': 'FirstChargeDensityLimit',
                    'FirstElectrodesDead (frac)': 'FirstDeactivationRate',
                    'FirstFalsePositiveRate': 'FirstFalsePositiveRate',
                    'TimeSinceFirstMeasurement (days)': 'TimeSinceFirstMeasurement',
                    'LastImpedance (kΩ)': 'LastImpedance',
                    'LastThresholds (µA)': 'LastThresholds',
                    'TimeSinceLastElectrodeMeasurement (days)': 'TimeSinceLastMeasurement'}

def get_adjusted_r2(r2, n_samples, n_predictors):
    r2_adjusted = 1-(((1-r2)*(n_samples-1))/(n_samples-n_predictors-1))
    return r2_adjusted