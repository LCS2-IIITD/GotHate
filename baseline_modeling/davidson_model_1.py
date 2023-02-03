import argparse
import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.stem.porter import *
import string
import re
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from datetime import datetime
import warnings
import time

warnings.simplefilter(action='ignore', category=FutureWarning)
import gc
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from spacy.lang.hi import STOP_WORDS as STOP_WORDS_HI

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    os.environ["OMP_NUM_THREADS"] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text


def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]+", tweet.lower())).strip()
    tweet = tweet.lower().strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens
    return tweet.split()


def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]+", tweet.lower())).strip()
    tweet = tweet.lower().strip()
    return tweet.split()


def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    
    Returns counts of urls, mentions, and hashtags.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return (parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'),
            parsed_text.count('HASHTAGHERE'))


def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    sentiment = sentiment_analyzer.polarity_scores(tweet)

    words = preprocess(tweet)  #Get text only

    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)
    num_unique_terms = len(set(words.split()))

    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(
        float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59,
        1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(
        206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)),
        2)

    twitter_objs = count_twitter_objs(tweet)
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [
        FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms,
        num_words, num_unique_terms, sentiment['neg'], sentiment['pos'],
        sentiment['neu'], sentiment['compound'], twitter_objs[2],
        twitter_objs[1], twitter_objs[0], retweet
    ]
    #features = pandas.DataFrame(features)
    return features


def get_feature_array(tweets):
    feats = []
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)
    
    
def get_vectorisers(train_tweets):
    global vectorizer_david, pos_vectorizer_david 
    vectorizer_david = vectorizer_david.fit(train_tweets)
    train_tweet_tags = get_tag_list(train_tweets)
    pos_vectorizer_david = pos_vectorizer_david.fit(
        pd.Series(train_tweet_tags))


def reset_vectorisers():
    global vectorizer_david, pos_vectorizer_david
    vectorizer_david = TfidfVectorizer(tokenizer=tokenize,
                                       preprocessor=preprocess,
                                       ngram_range=(1, 3),
                                       stop_words=stopwords,
                                       use_idf=True,
                                       smooth_idf=False,
                                       norm=None,
                                       decode_error='replace',
                                       max_features=5000,
                                       min_df=5,
                                       max_df=0.75)

    pos_vectorizer_david = TfidfVectorizer(
        tokenizer=None,
        lowercase=False,
        preprocessor=None,
        ngram_range=(1, 3),
        stop_words=None,
        use_idf=False,
        smooth_idf=False,
        norm=None,
        decode_error='replace',
        max_features=5000,
        min_df=5,
        max_df=0.75,
    )


def get_tag_list(tweets):
    tweet_tags = []
    for tweet in tweets:
        tokens = basic_tokenize(preprocess(tweet))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    return tweet_tags


def return_feature_set(tweets):
    global vectorizer_david, pos_vectorizer_david
    tfidf = vectorizer_david.transform(tweets).toarray()
    tweet_tags = get_tag_list(tweets)
    pos = pos_vectorizer_david.transform(pd.Series(tweet_tags)).toarray()
    feats = get_feature_array(tweets)
    feat_M = np.concatenate([tfidf, pos, feats], axis=1)
    del tfidf, pos, feats, tweet_tags
    gc.collect()
    return feat_M


def run_model(train_texts, train_labels,class_weights=None):
    reset_vectorisers()
    get_vectorisers(train_texts)
    X_train = return_feature_set(train_texts)
    y_train = np.asarray(train_labels)
    if class_weights is not None:
        base_model = LogisticRegression(class_weight=class_weights)
    else:
         base_model = LogisticRegression()
    base_model.fit(X_train, y_train)
    print("TRAIN ACCURACY")
    y_preds = base_model.predict(X_train)
    report = classification_report(y_train, y_preds,digits=5)
    print(report)
    return base_model
    
    
def get_data(dataset_type="train",base="HOPN",binary=False,topic_exclude=[],final="_final"):
    print("BASE DATASET:: ",base)
    INPUT_PATH = "" ## ADD YOUR PATH HERE
    david_chhavi = pd.read_csv(os.path.join(INPUT_PATH+base,dataset_type+final+".csv"))
    print(david_chhavi.shape)
    if base == "HOPN":
        texts = list(david_chhavi['text'])
    else:
        texts = list(david_chhavi['clean_text'])
    labels =list(david_chhavi['label'])
    plt.hist(labels)
    plt.show()
    c = list(zip(texts, labels))
    random.shuffle(c)
    texts, labels = zip(*c)
    texts = list(texts)
    labels = list(labels)
    return texts, labels
    
    
if __name__=="__main__":
    start = datetime.now()
    print("\n\Classification started started at ", start, "\n\n")
    set_random_seed(42)
    stopwords = stopwords = nltk.corpus.stopwords.words("english")
    other_exclusions = ["#ff", "ff", "rt"]
    stopwords.extend(other_exclusions)
    stopwords = list(set(stopwords))
    stemmer = PorterStemmer()
    sentiment_analyzer = VS()
    
    vectorizer_david = TfidfVectorizer(tokenizer=tokenize,
                                   preprocessor=preprocess,
                                   ngram_range=(1, 3),
                                   stop_words=stopwords,
                                   use_idf=True,
                                   smooth_idf=False,
                                   norm=None,
                                   decode_error='replace',
                                   max_features=5000,
                                   min_df=5,
                                   max_df=0.75)

    pos_vectorizer_david = TfidfVectorizer(
        tokenizer=None,
        lowercase=False,
        preprocessor=None,
        ngram_range=(1, 3),
        stop_words=None,
        use_idf=False,
        smooth_idf=False,
        norm=None,
        decode_error='replace',
        max_features=5000,
        min_df=5,
        max_df=0.75,
    )

    sentiment_analyzer = VS()

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="HOPN") 
    args = parser.parse_args()
    
    BASE_TYPE = args.base
    
    print("\n\n---------------------")
    train_texts, train_labels = get_data("train",base=BASE_TYPE, binary=False,final="_final")
    val_texts, val_labels = get_data("val",base=BASE_TYPE, binary=False,final="_final")
    test_texts, test_labels = get_data("test",base=BASE_TYPE, binary=False,final="_final")
    
    class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(train_labels),y=train_labels)
    class_weights = dict(zip(np.unique(train_labels), class_weights))
    print("Class weights:: ", class_weights)
    
    
    model = run_model(train_texts, train_labels,class_weights=class_weights)
    print("\n VAL ACCS")
    val_texts_feat = return_feature_set(val_texts)
    y_pred = model.predict(val_texts_feat)
    print(classification_report(val_labels, y_pred,digits=5))
    print("\n TEST ACCS")
    test_texts_feat = return_feature_set(test_texts)
    y_pred = model.predict(test_texts_feat)
    print(classification_report(test_labels, y_pred,digits=5))
    
    end = datetime.now()
    print("\n\nPredictions ended at ", end)
    print("\n\nTotal time taken ", end-start)
