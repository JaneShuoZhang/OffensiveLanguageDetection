import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

def calculate_result(filename):
    results = pd.read_csv(filename)
    result_summary = classification_report(results["subtask_a"], results["prediction"], output_dict=True)
    experiment_result = pd.DataFrame({'accuracy': [result_summary['accuracy']],
                                      'macro_precision': [result_summary["macro avg"]["precision"]],
                                      'macro_recall': [result_summary["macro avg"]["recall"]],
                                      'macro_F1': [result_summary["macro avg"]["f1-score"]],
                                      'micro_precision': [result_summary["weighted avg"]["precision"]],
                                      'micro_recall': [result_summary["weighted avg"]["recall"]],
                                      'micro_F1': [result_summary["weighted avg"]["f1-score"]]})
    return experiment_result

if __name__ == '__main__':
    RESULT_FOLDER = "result"
    experiment = pd.DataFrame({'method': [], 'accuracy': [],
                               'macro_precision': [], 'macro_recall': [], 'macro_F1': [],
                               'micro_precision': [], 'micro_recall': [], 'micro_F1': []})
    for dirName, subdirList, fileList in os.walk(RESULT_FOLDER):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            if fname.endswith(".csv"):
                print('\t%s' % fname)
                experiment_result = calculate_result(os.path.join(RESULT_FOLDER, fname))
                experiment_result['method'] = np.array([fname[:-4]])
                experiment = pd.concat([experiment, experiment_result], ignore_index=True)

    experiment.to_csv(os.path.join(RESULT_FOLDER, "experiment_compare.csv"), index=False)