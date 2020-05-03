import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

def calculate_result(filename):
    results = pd.read_csv(filename)
    result_summary = classification_report(results["subtask_a"], results["prediction"], output_dict=True)
    experiment_result = pd.DataFrame({'accuracy': [result_summary['accuracy']],
                                      'OFF_F1': [result_summary["OFF"]["f1-score"]],
                                      'macro_precision': [result_summary["macro avg"]["precision"]],
                                      'macro_recall': [result_summary["macro avg"]["recall"]],
                                      'macro_F1': [result_summary["macro avg"]["f1-score"]],
                                      'micro_precision': [result_summary["weighted avg"]["precision"]],
                                      'micro_recall': [result_summary["weighted avg"]["recall"]],
                                      'micro_F1': [result_summary["weighted avg"]["f1-score"]]})
    return experiment_result

def plot_confustion_matrix(filename):
    results = pd.read_csv(filename)
    confusion = pd.DataFrame(confusion_matrix(results["subtask_a"], results["prediction"]),
                        index=["NOT", "OFF"],
                        columns=["NOT", "OFF"])
    plt_save_path = filename[:-4] + '.png'
    sn.heatmap(confusion, cmap="YlGnBu", annot=True, fmt="d")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title("Confustion Matrix")
    plt.savefig(plt_save_path)

def compare_results(RESULT_FOLDER):
    experiment = pd.DataFrame({'method': [], 'accuracy': [], 'OFF_F1': [],
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

if __name__ == '__main__':
    RESULT_FOLDER = "result"

    compare_results(RESULT_FOLDER)

    #plot_confustion_matrix(os.path.join(RESULT_FOLDER, 'BERT_Iter_3_prediction.csv'))