import numpy as np
import pandas as pd
import os
import random
import sys

OILD_DATA_FOLDER = "../data/OLIDv1"
TRAIN_DATA_FILE = "olid-training-v1.tsv"
TEST_DATA_A_FILE = "testset-levela.tsv"
TEST_LABEL_A_FILE = "labels-levela.csv"
TRIAL_DATA_FILE = "trial-data/offenseval-trial.txt"

RESULT_FOLDER = "result"


def load_train_data():
    """ Load OLID training data. Only remain two columns: tweet, subtask_a (label).
    """
    training_data_file_path = os.path.join(OILD_DATA_FOLDER, TRAIN_DATA_FILE)
    train_data = pd.read_csv(training_data_file_path, sep='\t')
    train_data.drop(['id', 'subtask_b', 'subtask_c'], axis=1, inplace=True)
    print("number of training data: {}".format(train_data.shape[0]))
    return train_data


def load_test_data_a():
    """ Load subtask A test data. Only remain two columns: tweet, subtask_a (label).
    """
    test_data_file_path = os.path.join(OILD_DATA_FOLDER, TEST_DATA_A_FILE)
    test_data = pd.read_csv(test_data_file_path, sep='\t')
    test_label_file_path = os.path.join(OILD_DATA_FOLDER, TEST_LABEL_A_FILE)
    test_labels = pd.read_csv(test_label_file_path, header=None, names=['id', 'subtask_a'])
    test_data = test_data.join(test_labels.set_index('id'), on='id')
    test_data.drop(['id'], axis=1, inplace=True)
    print("number of test data A: {}".format(test_data.shape[0]))
    return test_data


def load_trial_data():
    """ Load OLID trial data. Only remain two columns: tweet, subtask_a (label).
    """
    trial_data_file_path = os.path.join(OILD_DATA_FOLDER, TRIAL_DATA_FILE)
    trial_data = pd.read_csv(trial_data_file_path, sep='\t', header=None, names=['tweet', 'subtask_a', 'subtask_b', 'subtask_c'])
    trial_data.drop(['subtask_b', 'subtask_c'], axis=1, inplace=True)
    print("number of trial data: {}".format(trial_data.shape[0]))
    return trial_data


def glove2dict(src_filename, dim=300):
    """GloVe Reader.
    Parameters
    ----------
    dim : int
        Full path to the GloVe file to be processed.
    Returns
    -------
    dict
        Mapping words to their GloVe vectors as `np.array`.
    """
    # This distribution has some words with spaces, so we have to
    # assume its dimensionality and parse out the lines specially:
    if '.300d' in src_filename:
        line_parser = lambda line: line.rsplit(" ", 300)
    else:
        line_parser = lambda line: line.strip().split()
    data = {}
    with open(src_filename, encoding='utf8') as f:
        while True:
            try:
                line = next(f)
                line = line_parser(line)
                if len(line) == dim+1:
                    data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    f.close()
    print("GloVe loaded. Vocabulary size: {}".format(len(data)))
    return data


def randvec(n=50, lower=-0.5, upper=0.5):
    """Returns a random vector of length `n`. `w` is ignored.
    """
    return np.array([random.uniform(lower, upper) for i in range(n)])

def progress_bar(msg):
    """Simple over-writing progress bar."""
    sys.stderr.write('\r')
    sys.stderr.write(msg)
    sys.stderr.flush()

def change_to_binary(label):
    return [0 if(i == 'OFF') else 1 for i in label]