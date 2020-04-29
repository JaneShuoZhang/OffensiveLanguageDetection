import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from nltk.util import ngrams
from utils import glove2dict, randvec

EMBEDDING_FOLDER = "../embedding"
GLOVE_EMBEDDING_FILE = {25: "glove.twitter.27B.25d.txt",
                        50: "glove.twitter.27B.50d.txt",
                        100: "glove.twitter.27B.100d.txt",
                        200: "glove.twitter.27B.200d.txt",
                        300: "glove.42B.300d.txt"}

def get_character_ngrams_of_word(word, n):
    if n == 1:
        return list(word)
    grams = list(ngrams(list(word), n))
    grams = [''.join(gram) for gram in grams]
    return grams


def get_all_character_ngrams_of_sentence(sentence, n=4):
    grams = []
    tokens = sentence.split()
    for word in tokens:
        for i in range(1, min(n, len(word))+1):
            grams.extend(get_character_ngrams_of_word(word, i))
    gram_dict = dict(Counter(grams))
    return gram_dict


def build_ngrams_dataset(df, n=4, vectorizer=None):
    """Core general function for building experimental datasets based on character n-grams feature.
    
    Parameters
    ----------
    df : DataFrame 
        Pre-processed dataset.
    n : int 
        Up to n grams.
    vectorizer : sklearn.feature_extraction.DictVectorizer 
        If this is None, then a new 'DictVectorizer' is created and used to 
        turn the list of dicts created by get_all_character_ngrams_of_sentence
        into a feature matrix. This happens when we are training.
        If this is not None, then it's assumed to be a 'DictVectorizer' 
        and used to transform the list of dicts. This happens in assessment, 
        when we take in new instances and need to featurize them as we did in training.
        
    Returns
    -------
    dict
        A dict with keys 'X' (the feature matrix), 'y' (the list of labels),
        'vectorizer' (the 'DictVectorizer' object).
    """
    feature_dicts = []
    labels = []
    
    for index, row in df.iterrows():
        feature_dicts.append(get_all_character_ngrams_of_sentence(row['tweet']))
        encoded_label = 1 if row['subtask_a'] == 'OFF' else 0
        labels.append(encoded_label)
        
    if vectorizer == None:
        vectorizer = DictVectorizer(sparse=False)
        feature_matrix = vectorizer.fit_transform(feature_dicts)
    else:
        feature_matrix = vectorizer.transform(feature_dicts)
        
    return {'X': feature_matrix,
            'y': labels,
            'vectorizer': vectorizer}


def glove_featurizer(tweet, glove_lookup, np_func=np.sum):
    """Get vector representation of one tweet.
    """
    reps = []
    tokens = tweet.split()
    for word in tokens:
        rep = glove_lookup.get(word)
        if rep is not None:
            reps.append(rep)
    # A random representation of the right dimensionality if the
    # example happens not to overlap with GloVe's vocabulary:
    if len(reps) == 0:
        dim = len(next(iter(glove_lookup.values())))                
        return randvec(n=dim)
    else:
        return np_func(reps, axis=0)


def build_glove_featurized_dataset(df, dim=300, np_func=np.sum):
    if dim not in GLOVE_EMBEDDING_FILE:
        print("GloVe file of dim {} is not found.".format(dim))
        return None
    
    glove_file_path = os.path.join(EMBEDDING_FOLDER, GLOVE_EMBEDDING_FILE[dim])
    glove_lookup = glove2dict(glove_file_path, dim)

    feature_matrix = []
    labels = []

    for index, row in df.iterrows():
        feature_matrix.append(glove_featurizer(row['tweet'], glove_lookup, np_func))
        encoded_label = 1 if row['subtask_a'] == 'OFF' else 0
        labels.append(encoded_label)

    return {'X': np.array(feature_matrix),
            'y': labels}


def generate_glove_embedding(dim=300):
    if dim not in GLOVE_EMBEDDING_FILE:
        print("GloVe file of dim {} is not found.".format(dim))
        return None

    glove_file_path = os.path.join(EMBEDDING_FOLDER, GLOVE_EMBEDDING_FILE[dim])
    glove_lookup = glove2dict(glove_file_path, dim)

    vocal = ['$UNK']
    embedding = [list(np.random.uniform(low=-1.0, high=1.0, size=dim))]
    for key, value in glove_lookup.items():
        vocal.append(key)
        embedding.append(list(value))

    embedding = np.array(embedding, dtype=np.float)
    print("Vocabulary size: {}, Embedding size{}".format(len(vocal), embedding.shape))
    return [vocal, embedding]


def build_LSTM_dataset(df, max_seq_length):
    feature_matrix = []
    labels = []

    for index, row in df.iterrows():
        feature = row['tweet'].split()
        feature_matrix.append(feature[:max_seq_length])
        labels.append(row['subtask_a'])
    return [feature_matrix, labels]