import numpy as np
import pickle
import argparse
from sklearn.metrics import r2_score
from analysis_utils import subjects, get_model_results, get_adjusted_r2


parser = argparse.ArgumentParser()
parser.add_argument('--results_file', \
                    type=str, \
                    required=True, \
                    help='Path to saved model artifact file')
args = parser.parse_args()
if __name__ == '__main__':
    results_file = args.results_file
    X_tests, y_tests, y_hats = get_model_results(results_file)
    subjects = X_tests.keys()

    print('Aggregate Data Results:')
    X_test_all = np.vstack([X_tests[subject] for subject in subjects])
    y_test_all = np.hstack([y_tests[subject] for subject in subjects])
    y_hat_all = np.hstack([y_hats[subject] for subject in subjects])
    
    r2 = r2_score(y_test_all, y_hat_all)
    n, k = X_test_all.shape
    r2_adjusted = get_adjusted_r2(r2, n, k)
    print('\tAdjusted R2:\t{:.4f}\n'.format(r2_adjusted))
    
    print('Subject Results:')
    r2_adjusted_subs = []
    for i, subject in enumerate(subjects):
        r2 = r2_score(y_tests[subject], y_hats[subject])
        n, k = X_tests[subject].shape
        r2_adjusted_subs.append(get_adjusted_r2(r2, n, k))
        print('\tSubject {} Adjusted R2: {:.4f}'.format(subject, r2_adjusted_subs[-1]))
    print()
    print('\tMean +/- Std. Dev.: {:.4f} +/- {:.4f}'.format(np.mean(r2_adjusted_subs), np.std(r2_adjusted_subs)))