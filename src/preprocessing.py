import pandas as pd
import os
import re
import time
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from wordsegment import load, segment
import emoji
from utils import load_train_data, load_test_data_a, load_trial_data
import symspell_python as spell_checkers

pd.options.mode.chained_assignment = None  # default='warn'

PROCESSED_DATA_FOLDER = "processed_data"
PROCESSED_TRAIN_DATA_FILE = "train.csv"
PROCESSED_TEST_DATA_FILE = "test_a.csv"
PROCESSED_TRIAL_DATA_FILE = "trial.csv"

cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expand_contractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

def segment_word(word):
    if '!' in word or '?' in word or '.' in word:
        return [word]
    return segment(word)


def process_tweets(tweets_column):
    # Replace Emoji by substituted phrase
    tweets_column = tweets_column.apply(lambda x: emoji.demojize(x, delimiters=(',', ',')))
    # Lowercase tweets
    tweets_column = tweets_column.apply(lambda x: x.lower())
    # Apostrophe expansion
    tweets_column = tweets_column.apply(lambda x: x.replace("’", "'"))
    tweets_column = tweets_column.apply(lambda x: expand_contractions(x))
    # Remove url, hashtags, cashtags, twitter handles, and RT. Only words, numbers, !, ? and .
    tweets_column = tweets_column.apply(
        lambda x: ' '.join(re.sub(r"(@[A-Za-z]+)|^rt |(\w+:\/*\S+)|[^a-zA-Z0-9\s!?.]", "", x).split()))
    # Remove url token
    tweets_column = tweets_column.apply(lambda x: x.replace('url', ''))

    # Word segmentation.
    load()
    tweets_column = tweets_column.apply(lambda x: [segment_word(word) for word in x.split()])
    # flatten
    tweets_column = tweets_column.apply(lambda x: ' '.join([item for splitted in x for item in splitted]))

    # Tokenize
    tokeniser = TweetTokenizer()
    tweets_column = tweets_column.apply(lambda x: [word for word in tokeniser.tokenize(x)])

    spell_checkers.create_dictionary("eng_dict.txt")
    print("Spell checker...")
    for i in range(len(tweets_column)):
        try:
            # print('%i out of %i' % (i, len(tweets_column)))
            for j in range(len(tweets_column[i])):
                suggs = spell_checkers.get_suggestions(tweets_column[i][j])
                if suggs:
                    best_sugg = str(suggs[0])
                    tweets_column[i][j] = best_sugg
        except:
            continue

    # Lemmatisation
    wordnet_lemmatizer = WordNetLemmatizer()
    tweets_column = tweets_column.apply(lambda x: ' '.join([wordnet_lemmatizer.lemmatize(word, pos="v") for word in x]))

    return tweets_column


def process_single_tweet(tweet, lemmatize=True):
    """ Process and tokenize single tweet.
    """        
    # Replace Emoji by substituted phrase
    tweet = emoji.demojize(tweet, delimiters=(',', ','))
    
    # Lowercase tweets
    tweet = tweet.lower()
    
    # Apostrophe expansion
    tweet = tweet.replace("’","'")
    tweet = expand_contractions(tweet)   
    
    # Remove twitter handles, RT, url. Remain only letters, numbers, !, ? and .
    tweet = ' '.join(re.sub( \
    r"(@[A-Za-z]+)|^rt |(\w+:\/*\S+)|[^a-zA-Z0-9\s!?.]", "" ,tweet).split())
    
    # Word segmentation.
    load()
    splitted_tweet = []
    [splitted_tweet.extend(segment_word(word)) for word in tweet.split()]
    tweet = ' '.join(splitted_tweet)
    
    # Remove url token
    tweet = tweet.replace('url','')

    # Tokenize
    twt_tokenizer = TweetTokenizer()
    tokens = twt_tokenizer.tokenize(tweet)

    # Spell check
    spell_checkers.create_dictionary("eng_dict.txt")
    try:
        for i in range(len(tokens)):
                suggs = spell_checkers.get_suggestions(tokens[i])
                if suggs:
                    best_sugg = str(suggs[0])
                    tokens[i] = best_sugg
    except:
        print("Warning in spell checking.")
    
    # Lemmatize
    if lemmatize:
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = [wordnet_lemmatizer.lemmatize(word, pos = "v") for word in tokens]

    tweet = ' '.join(tokens)
    return tweet


def process_data(df, output_file_path, replace=False):
    """Pre-process dataset. If processed data already exist in folder, 
    directly load the data instead of processing again, if replace == False.

    Parameters
    ----------
    df : DataFrame
        The dataset.
    output_file_path : string 
        Path of outputed processed data.
    replace : bool
        Whether to replace the existing processed data if the data was processed before.

    Returns
    -------
    DataFrame
        The pre-processed data.
    """
    if not replace and os.path.exists(output_file_path):
        df = pd.read_csv(output_file_path)
        print("Processed data already exists. Direct load it.")
        print("number of processed data: {}".format(df.shape[0]))
        return df
    else:        
        if not os.path.exists(PROCESSED_DATA_FOLDER):
            os.makedirs(PROCESSED_DATA_FOLDER)
        start_time = time.time()
        df.dropna()
        df.rename(columns={'tweet': 'raw_tweet'}, inplace=True)
        #df['tweet'] = df['raw_tweet'].apply(process_single_tweet)
        df['tweet'] = process_tweets(df['raw_tweet'])
        df.drop(['raw_tweet'], axis=1, inplace=True)
        df.to_csv(output_file_path, index=False)
        print("number of  processed data: {}".format(df.shape[0]))
        end_time = time.time()
        print("Process data in {} mins.".format((end_time - start_time)/60))
        return df


def process_train_data(df, replace=False):
    """Pre-process training data.
    """
    output_file_path = os.path.join(PROCESSED_DATA_FOLDER, PROCESSED_TRAIN_DATA_FILE)
    return process_data(df, output_file_path, replace)


def process_test_data(df, replace=False):
    """Pre-process test data.
    """
    output_file_path = os.path.join(PROCESSED_DATA_FOLDER, PROCESSED_TEST_DATA_FILE)
    return process_data(df, output_file_path, replace)


def process_trial_data(df, replace=False):
    """Pre-process trial data.
    """
    output_file_path = os.path.join(PROCESSED_DATA_FOLDER, PROCESSED_TRIAL_DATA_FILE)
    return process_data(df, output_file_path, replace)



if __name__ == "__main__":
    print("Start Processing Trial Data:")
    trial_data = load_trial_data()
    process_trial_data(trial_data)

    print("Start Processing Test Data:")
    test_data = load_test_data_a()
    process_test_data(test_data)

    print("Start Processing Train Data:")
    train_data = load_train_data()
    process_train_data(train_data)