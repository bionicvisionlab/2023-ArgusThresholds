[![Build Status](https://github.com/pulse2percept/pulse2percept/workflows/build/badge.svg)](https://github.com/pulse2percept/pulse2percept/actions)
[![Coverage Status](https://coveralls.io/repos/github/bionicvisionlab/ArgusThresholds/badge.svg?branch=master)](https://coveralls.io/github/bionicvisionlab/ArgusThresholds?branch=master)


## ArgusThresholds

Predicting Argus II thresholds from a variety of clinical and physiological factors.

To provide appropriate levels of stimulation, retinal prostheses must be calibrated to an individual's
perceptual thresholds, despite thresholds varying drastically across subjects, across electrodes within a
subject, and over time.

Although previous work has identified electrode-retina distance and impedance as key
factors affecting thresholds, an accurate predictive model is still lacking.

The aim of this study is thus to develop a model that can

1. predict thresholds on individual electrodes as a function of stimulus, electrode, and clinical parameters (‘predictors’), and
2. reveal which of these predictors are most important.

### Installation

```
git clone https://github.com/bionicvisionlab/ArgusThresholds.git
cd ArgusThresholds
pip install -e .
```

### Getting Started

1. Download the data. Currently it is assumed that all research data lives in a local directory pointed to
   by the environment variable `$DATA_ROOT`, and that there's a sub-directory called "argus_thresholds"
   where all the data for this study live.
   On DeepThought, `$DATA_ROOT` should be set to `/usr/data`.

2. You can use `argus_thresholds.load_data` to load data from all subjects from the various spreadsheets.

3. You can preprocess the data with `argus_thresholds.preprocess_data`. Have a look at the function, it
   does a number of interesting things to remove electrodes and subjects for which there aren't enough
   data points.
   On DeepThought, the preprocessed data is already stored in `/usr/data/argus_thresholds`.

#### Scripts

Scripts can be found in the `scripts/` folder:

* `python predict-elastic-2020.py <mode> <normalize>`: Hyperparameter tuning for the ElasticNet baseline.
  Specify a mode to select a subset of feature columns ('original', 'clinical', 'image', etc.) and
  whether to normalize the data before fitting (1: normalize, 0: don't normalize)

* `python predict-tree-2020.py <mode>`: Hyperparameter tuning for XGBoost.
  Specify a mode as described above.

### Notes

*  Please work on your own branch or private fork.
*  Commit early and often. It helps us keep track and provides you with a backup copy of your latest code.
*  Please don't share the data or put it on Dropbox/Box/Google Drive/etc. We don't have permission for that.
