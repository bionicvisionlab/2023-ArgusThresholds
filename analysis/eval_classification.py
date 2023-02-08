import numpy as np
import pickle
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score
from analysis_utils import subjects, get_model_results, get_auc


parser = argparse.ArgumentParser()
parser.add_argument('--artifact_dir', \
                    type=str, \
                    required=True, \
                    help='Directory containing fitted model files')
parser.add_argument('--results_file', \
                    type=str, \
                    required=True, \
                    help='Path to saved model artifact file')
args = parser.parse_args()
if __name__ == '__main__':
    artifact_dir = args.artifact_dir
    results_file = args.results_file
    X_tests, y_tests, y_hats = get_model_results(results_file)
    subjects = X_tests.keys()

    print('Aggregate Data Results:')
    X_test_all = np.vstack([X_tests[subject] for subject in subjects])
    y_test_all = np.hstack([y_tests[subject] for subject in subjects])
    y_hat_all = np.hstack([y_hats[subject] for subject in subjects])
    
    precision = precision_score(y_test_all, y_hat_all)
    recall = recall_score(y_test_all, y_hat_all)
    f1 = f1_score(y_test_all, y_hat_all)
    if np.sum(y_test_all) > 1:
        auc = get_auc(artifact_dir, results_file, subjects)
    else:
        auc = np.nan
    print('\tPrecision: {:.4f}'.format(precision))
    print('\tRecall: {:.4f}'.format(recall))
    print('\tF1: {:.4f}'.format(f1))
    print('\tAUC: {:.4f}\n'.format(auc))
    
    print('Subject Results:')
    precision_subs = []
    recall_subs = []
    f1_subs = []
    auc_subs = []
    for i, subject in enumerate(subjects):
        print('\tSubject {}'.format(subject))
        if np.sum(y_tests[subject]) == 0:
            print('\t\tNo positive samples\n')
            continue
        precision_subs.append(precision_score(y_tests[subject], y_hats[subject]))
        recall_subs.append(recall_score(y_tests[subject], y_hats[subject]))
        f1_subs.append(f1_score(y_tests[subject], y_hats[subject]))
        auc_subs.append(get_auc(artifact_dir, results_file, [subject]))
        print('\t\tPrecision: {:.4f}'.format(precision_subs[-1]))
        print('\t\tRecall: {:.4f}'.format(recall_subs[-1]))
        print('\t\tF1: {:.4f}'.format(f1_subs[-1]))
        print('\t\tAUC: {:.4f}\n'.format(auc_subs[-1]))
    
    print('\tPrecision Mean +/- Std. Dev.: {:.4f} +/- {:.4f}'\
          .format(np.mean(precision_subs), np.std(precision_subs)))
    print('\tRecall Mean +/- Std. Dev.: {:.4f} +/- {:.4f}'\
          .format(np.mean(recall_subs), np.std(recall_subs)))
    print('\tF1 Mean +/- Std. Dev.: {:.4f} +/- {:.4f}'\
          .format(np.mean(f1_subs), np.std(f1_subs)))
    print('\tAUC Mean +/- Std. Dev.: {:.4f} +/- {:.4f}'\
          .format(np.mean(auc_subs), np.std(auc_subs)))