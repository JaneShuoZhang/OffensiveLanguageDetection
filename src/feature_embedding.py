import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from nltk.util import ngrams


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