## ArgusThresholds

Predicting Argus II thresholds from a variety of clinical and physiological factors.

To provide appropriate levels of stimulation, retinal prostheses must be calibrated to an individual’s
perceptual thresholds, despite thresholds varying drastically across subjects, across electrodes within a subject, and over time. 

Although previous work has identified electrode-retina distance and impedance as key factors affecting thresholds, an accurate predictive model is still lacking.

To address these challenges, we set out to develop explainable machine learning (ML) models that could:
- predict perceptual thresholds on individual electrodes as a function of stimulus, electrode, and clinical parameters
- infer deactivation of individual electrodes as a function of these parameters, and
- reveal which of these predictors were most important to perceptual thresholds and electrode deactivation

This repository contains the implementation of the work presented in [Explainable Machine Learning Predictions of
Perceptual Sensitivity for Retinal Prostheses](https://github.com/bionicvisionlab/2023-ArgusThresholds), where these challenges were studied.

### Installation

```
git clone https://github.com/bionicvisionlab/2023-ArgusThresholds.git
cd 2023-ArgusThresholds
pip install -e .
```

### Scripts

Scripts can be found in the `scripts/` folder:

* `python preprocess-store-data.py <mode> <scale_thresholds> <ignore_outliers> <keep_nans> <min_samples>`: Data preprocessing and feature extraction for downstream classification and regression tasks.
   * `mode` should be one of ['routine', 'fitting', 'followup']
   * For classification tasks: 
      * `ignore_outliers` = False
      * `keep_nans` = False
      * `min_samples` = 50
   * For for regression tasks:
      * `ignore_outliers` = True
      * `keep_nans` = False
      * `min_samples` = 50
      * `scale_thresholds` = 'first' if and only if `mode` is 'fitting' or 'followup'
   
* `python fit-predict-classification-2023.py <mode> <datapath> <model> <standardize> <upsample> <paramsearch> <outpath>`: Model fitting for electrode deactivation classification.
   * `mode`: one of ['routine', 'fitting', 'followup']
   * `datapath`: Path to preprocessed data associated with selected mode
   * `model`: 'logreg' or 'xgb'
   * `standardize`: Standardize feature values (set to True in all experiments)
   * `upsample`: Upsample minority class with SMOTE (set to True in all experiments)
   * `paramsearch`: Set to True to run Bayesian hyperparameter optimization
   * `outpath`: Path to save trained models and associated artifacts
   
* `python fit-predict-regression-2023.py <mode> <datapath> <model> <standardize> <upsample> <paramsearch> <outpath>`: Model fitting for perceptual threshold regression.
   * `mode`: one of ['routine', 'fitting', 'followup']
   * `datapath`: Path to preprocessed data associated with selected mode
   * `model`: 'elasticnet' or 'xgb'
   * `standardize`: Standardize feature values (set to True in all experiments)
   * `paramsearch`: Set to True to run Bayesian hyperparameter optimization
   * `outpath`: Path to save trained models and associated artifacts

### Analyses

Code supporting result analysis and figure generation can be found in the `analysis/` folder:

* `python plot_correlation_heatmap.py`: Plotting feature correlation heatmap (Figure 1).
* `python plot_spearman_correlations.py`: Plotting feature correlation heatmap (Figure 3).
* `python plot_features.py`: Scatterplots between features and perceptual thresholds  (Figure 4).
* `python plot_shap.py`: Plotting top 10 features according to SHAP values (Figures 5, 6).
* `python plot_threshold_kdes.py`: Plotting threshold KDEs (Appendix Figure 1).
* `python plot_regression_results.py`: Regression ground-truth vs. prediction scatterplots (Appendix Figures 2, 3).
* `python eval_regression.py`: Runs evaluation of perceptural threshold prediction.
* `python eval_classification.py`: Runs evaluation of electrode classification deactivation.
  
  
## Reference
[1] Galen Pogoncheff, Zuying Hu, Ariel Rokem, and Michael Beyeler.  Explainable Machine Learning Predictions of Perceptual Sensitivity for Retinal Prostheses.
