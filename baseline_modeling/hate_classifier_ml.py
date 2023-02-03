# Code for using ML (SVM and Naive Bayes) algorithms for hate classification
# File: hate_classifier_ml.py
# Author: Atharva Kulkarni

# nohup python3 -u hate-classifier-ml.py --model <MultinomialNB/ComplementNB/LinearSVC/LinearSVC-weighted/LR/LR-weighted> 


import argparse
import os
import numpy as np
import pandas as pd
import pickle
import warnings
import random
import gc
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)
from sklearn.metrics import (
    f1_score, 
    recall_score, 
    precision_score, 
    accuracy_score,
    classification_report
)

from sklearn.svm import (
    LinearSVC,
    NuSVC
)
from sklearn.naive_bayes import (
    ComplementNB,
    MultinomialNB
)

from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


# -------------------------------------------------------------- CONFIG -------------------------------------------------------------- #

INPUT_PATH = '' ## ADD YOUR PATH HERE
OUTPUT_PATH = '' ## ADD YOUR PATH HERE


if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("Using GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")



LEMMATIZER = WordNetLemmatizer()

STOPWORDS = stopwords.words('english')
unwanted_stopwords = {'no', 'nor', 'not', 'ain', 'aren', "aren't", 'couldn', 'what', 'which', 'who','whom','why', 'how',
                      "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",'hasn',"hasn't", 'haven', "haven't", 
                      'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',"mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                      "shouldn't", 'wasn',"wasn't",'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'don', "don't"}

STOPWORDS = [ele for ele in STOPWORDS if ele not in unwanted_stopwords]


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


SEED = 42
set_random_seed(SEED)


# -------------------------------------------------------------- UTILS -------------------------------------------------------------- #


def preprocess_text(user_text):
    # Remove puntuations and numbers
    user_text = re.sub('[^a-zA-Z]', ' ', user_text)

    # Remove single characters
    user_text = re.sub(r"\s+[a-zA-Z]\s+", ' ', user_text)

    # remove multiple spaces
    user_text = re.sub(r'\s+', ' ', user_text)
    user_text = user_text.lower()

    # Convert Text sentence to Tokens
    user_text = word_tokenize(user_text)

    # Remove unncecessay stopwords
    fintered_text = []
    for t in user_text:
        if t not in STOPWORDS:
            fintered_text.append(t)

    # Word lemmatization
    processed_text1 = []
    for t in fintered_text:
        word1 = LEMMATIZER.lemmatize(t, pos="n")
        word2 = LEMMATIZER.lemmatize(word1, pos="v")
        word3 = LEMMATIZER.lemmatize(word2, pos=("a"))
        processed_text1.append(word3)

    result = ""
    for word in processed_text1:
        result = result + word + " "
    result = result.rstrip()

    return result



def count_vectorize(X_train, X_val, X_test):
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=50000)
    count_vect.fit(X_train)

    # transform the training and validation data using count vectorizer object
    xtrain_count = count_vect.transform(X_train)
    xval_count = count_vect.transform(X_val)
    xtest_count = count_vect.transform(X_test)

    return xtrain_count, xval_count, xtest_count



def word_TF_IDF_vectorize(X_train, X_val, X_test):
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=50000)
    tfidf_vect.fit(X_train)

    xtrain_tfidf = tfidf_vect.transform(X_train)
    xval_tfidf = tfidf_vect.transform(X_val)
    xtest_tfidf = tfidf_vect.transform(X_test)

    return xtrain_tfidf, xval_tfidf, xtest_tfidf



def n_gram_TF_IDF_vectorize(X_train, X_val, X_test):
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=50000)
    tfidf_vect_ngram.fit(X_train)

    xtrain_tfidf_ngram = tfidf_vect_ngram.transform(X_train)
    xval_tfidf_ngram = tfidf_vect_ngram.transform(X_val)
    xtest_tfidf_ngram = tfidf_vect_ngram.transform(X_test)

    return xtrain_tfidf_ngram, xval_tfidf_ngram, xtest_tfidf_ngram



def char_TF_IDF_vectorize(X_train, X_val, X_test):
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char_wb', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=50000)
    tfidf_vect_ngram_chars.fit(X_train)

    xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(X_train)
    xval_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(X_val)
    xtest_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(X_test)

    return xtrain_tfidf_ngram_chars, xval_tfidf_ngram_chars, xtest_tfidf_ngram_chars



def get_prediction_scores(
    preds,
    gold
):
    return {
        'accuracy': accuracy_score(gold, preds),
        'precision': precision_score(gold, preds, average='macro'),
        'recall': recall_score(gold, preds, average='macro'),
        'f1-score': f1_score(gold, preds, average='macro')
    }, classification_report(y_true=gold, y_pred=preds, output_dict=False)


# ------------------------------------------------------------ MAIN MODEL ------------------------------------------------------------ #

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='MultinomialNB')
    args = parser.parse_args()
    
    # ------------------------------ READ DATASET ------------------------------ #
    
    train_df = pd.read_csv(INPUT_PATH+'train_final.csv')
    train_df['label'] = [0 if label == 3 else 1 for label in train_df['label']]
    print("\nLabel Distribution: \n", train_df['label'].value_counts())
    print("\n\nTopic Distribution: \n", train_df['topic'].value_counts())
#     train_data = train_df['clean_text'].apply(lambda x: preprocess_text(x)).tolist()
    train_data = train_df['clean_text'].values.tolist()
    train_labels = train_df['label'].tolist()
    print("\n\nTrain data size: {} label size: {}".format(len(train_data), len(train_labels)), "\n\n------------------------------------------------\n\n")

    val_df = pd.read_csv(INPUT_PATH+'val_final.csv')
    val_df['label'] = [0 if label == 3 else 1 for label in val_df['label']]
    print("\nLabel Distribution: \n", val_df['label'].value_counts())
    print("\n\nTopic Distribution: \n", val_df['topic'].value_counts())
#     val_data = val_df['clean_text'].apply(lambda x: preprocess_text(x)).tolist()
    val_data = val_df['clean_text'].values.tolist()
    val_labels = val_df['label'].tolist()
    print("\nVal data size: {} label size: {}".format(len(val_data), len(val_labels)), "\n\n------------------------------------------------\n\n")

    test_df = pd.read_csv(INPUT_PATH+'test_final.csv')
    test_df['label'] = [0 if label == 3 else 1 for label in test_df['label']]
    print("\nLabel Distribution: \n", test_df['label'].value_counts())
    print("\n\nTopic Distribution: \n", test_df['topic'].value_counts())
#     test_data = test_df['clean_text'].apply(lambda x: preprocess_text(x)).tolist()
    test_data = test_df['clean_text'].values.tolist()
    test_labels = test_df['label'].tolist()
    print("\nTest data size: {} label size: {}".format(len(test_data), len(test_labels)), "\n\n------------------------------------------------\n\n")
    
    # ------------------------------ PREPROCESS DATASET ------------------------------ #
    
    train_x_count, val_x_count, test_x_count = count_vectorize(train_data, val_data, test_data)
    print("\n\nCount processed train len: {}, val len: {}, test len: {}".format(train_x_count.shape, val_x_count.shape, test_x_count.shape))

    train_x_tfidf, val_x_tfidf, test_x_tfidf = word_TF_IDF_vectorize(train_data, val_data, test_data)
    print("\n\nTF-IDF processed train len: {}, val len: {}, test len: {}".format(train_x_tfidf.shape, val_x_tfidf.shape, test_x_tfidf.shape))

    train_x_ngram_tfidf, val_x_ngram_tfidf, test_x_ngram_tfidf = n_gram_TF_IDF_vectorize(train_data, val_data, test_data)
    print("\n\nN-gram TF-IDF processed train len: {}, val len: {}, test len: {}".format(train_x_ngram_tfidf.shape, val_x_ngram_tfidf.shape, test_x_ngram_tfidf.shape))

    train_x_char_tfidf, val_x_char_tfidf, test_x_char_tfidf = char_TF_IDF_vectorize(train_data, val_data, test_data)
    print("\n\nCharacter n-gram TF-IDF processed train len: {}, val len: {}, test len: {}".format(train_x_char_tfidf.shape, val_x_char_tfidf.shape, test_x_char_tfidf.shape))
    
    
    # ------------------------------ MODEL TRAINING ------------------------------ #
    
    if args.model == 'MultinomialNB':
        MODEL = MultinomialNB()
        print("\nMODEL: ", MODEL)
        
    elif args.model == 'ComplementNB':
        MODEL = ComplementNB()
        print("\nMODEL: ", MODEL)
        
    elif args.model == 'LinearSVC':
        MODEL = LinearSVC()
        print("\nMODEL: ", MODEL)
        
    elif args.model == 'LinearSVC-weighted':
        MODEL = LinearSVC(class_weight='balanced')
        print("\nMODEL: ", MODEL)
        
    elif args.model == 'LR':
        MODEL = LogisticRegression()
        print("\nMODEL: ", MODEL)
    
    elif args.model == 'LR-weighted':
        MODEL = LogisticRegression(class_weight='balanced')
        print("\nMODEL: ", MODEL)
    
    
    print("\n\n------------------------------------------------- Count -------------------------------------------------\n\n")
    MODEL.fit(train_x_count, train_labels)
    
    val_preds = MODEL.predict(val_x_count)
    val_results, val_cr = get_prediction_scores(val_preds, val_labels)
    print("\n\nval results: ", val_results)
    print("\n\nval classification report: \n", val_cr)
    print("\n\n--------------------------------------------------------------------------------------------------\n\n")
    
    test_preds = MODEL.predict(test_x_count)
    test_results, test_cr = get_prediction_scores(test_preds, test_labels)
    print("\n\ntest results: ", test_results)
    print("\n\ntest classification report: \n", test_cr, "\n\n")
    
    
    
    print("\n\n------------------------------------------------- TF-IDF -------------------------------------------------\n\n")
    MODEL.fit(train_x_tfidf, train_labels)
    
    val_preds = MODEL.predict(val_x_tfidf)
    val_results, val_cr = get_prediction_scores(val_preds, val_labels)
    print("val results: ", val_results)
    print("\n\nval classification report: \n", val_cr)
    print("\n\n--------------------------------------------------------------------------------------------------\n\n")
    
    test_preds = MODEL.predict(test_x_tfidf)
    test_results, test_cr = get_prediction_scores(test_preds, test_labels)
    print("\n\ntest results: ", test_results)
    print("\n\ntest classification report: \n", test_cr, "\n\n")
    
    
    
    print("\n\n------------------------------------------------- N-GRAM TF-IDF -------------------------------------------------\n\n")
    MODEL.fit(train_x_ngram_tfidf, train_labels)
    
    val_preds = MODEL.predict(val_x_ngram_tfidf)
    val_results, val_cr = get_prediction_scores(val_preds, val_labels)
    print("val results: ", val_results)
    print("\n\nval classification report: \n", val_cr)
    print("\n\n--------------------------------------------------------------------------------------------------\n\n")
    
    test_preds = MODEL.predict(test_x_ngram_tfidf)
    test_results, test_cr = get_prediction_scores(test_preds, test_labels)
    print("\n\ntest results: ", test_results)
    print("\n\ntest classification report: \n", test_cr, "\n\n")
    
    
    
    print("\n\n------------------------------------------------- character TF-IDF -------------------------------------------------\n\n")
    MODEL.fit(train_x_char_tfidf, train_labels)
    
    val_preds = MODEL.predict(val_x_char_tfidf)
    val_results, val_cr = get_prediction_scores(val_preds, val_labels)
    print("val results: ", val_results)
    print("\n\nval classification report: \n", val_cr)
    print("\n\n--------------------------------------------------------------------------------------------------\n\n")
    
    test_preds = MODEL.predict(test_x_char_tfidf)
    test_results, test_cr = get_prediction_scores(test_preds, test_labels)
    print("\n\ntest results: ", test_results)
    print("\n\ntest classification report: \n", test_cr, "\n\n")
    
