import argparse
import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.stem.porter import *
import string
import re
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import warnings
import time
import nltk
from nltk.corpus import stopwords

# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore")
import gc
import random

from spacy.lang.hi import STOP_WORDS as STOP_WORDS_HI
from spacy.lang.de import STOP_WORDS as STOP_WORDS_DE
from spacy.lang.es import STOP_WORDS as STOP_WORDS_ES

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
    text_string = text_string.strip().lower()
    space_pattern = '\s+'
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(mention_regex, ' ', parsed_text)
    text_string = re.sub(r'^https?:\/\/.*[\r\n]*', '', parsed_text, flags=re.MULTILINE) 
    parsed_text = parsed_text.translate(table)
    parsed_text = parsed_text.strip()
    return parsed_text


def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tokens = []
    for word in tweet.split():
        if word in STOP_LIST:
            continue
        tokens.append(word)
    return tokens
    
def get_vectorisers(train_tweets):
    global vectorizer_david
    vectorizer_david = vectorizer_david.fit(train_tweets)


def reset_vectorisers():
    global vectorizer_david
    vectorizer_david = TfidfVectorizer(
                                        tokenizer=tokenize,
                                       preprocessor=preprocess,
                                       ngram_range=(1, 3),
                                       stop_words=STOP_LIST,
                                       use_idf=True,
                                       smooth_idf=False,
                                       norm=None,
                                       decode_error='replace',
                                       max_features=10000,
                                       min_df=5,
                                       max_df=0.75)

def return_feature_set(tweets):
    global vectorizer_david
    tfidf = vectorizer_david.transform(tweets).toarray()
    return tfidf


def run_model(train_texts, train_labels,class_weights=None):
    reset_vectorisers()
    print("Gettting Vectorizer")
    get_vectorisers(train_texts)
    X_train = return_feature_set(train_texts)
    y_train = np.asarray(train_labels)
    if class_weights is not None:
        base_model = LogisticRegression(class_weight=class_weights)
    else:
         base_model = LogisticRegression()
    print("Fitting Model")
    base_model.fit(X_train, y_train)
    print("TRAIN ACCURACY")
    y_preds = base_model.predict(X_train)
    report = classification_report(y_train, y_preds)
    print(report)
    return base_model
    
    
def get_data():
    global BASE_TYPE, NEW_TYPE
    print("BASE DATASET:: ",BASE_TYPE)
    print("NEW DATASET:: ",NEW_TYPE)
    HOPN_label_map={0:0,1:0,2:0,3:1}
    
    INPUT_PATH = "" ## ADD YOUR PATH HERE
    train_data_old = pd.read_csv(INPUT_PATH+BASE_TYPE+"/train_final.csv",lineterminator="\n") 
    val_data_old = pd.read_csv(INPUT_PATH+BASE_TYPE+"/val_final.csv",lineterminator="\n")
#     test_data_old = pd.read_csv(INPUT_PATH+BASE_TYPE+"/test_final.csv",lineterminator="\n")
    print(train_data_old.shape,val_data_old.shape)
    print(train_data_old['label'].unique())
    
#     train_data_new = pd.read_csv(INPUT_PATH+NEW_TYPE+"/train_final.csv",lineterminator="\n")
#     val_data_new = pd.read_csv(INPUT_PATH+NEW_TYPE+"/val_final.csv",lineterminator="\n")
    test_data_new = pd.read_csv(INPUT_PATH+NEW_TYPE+"/test_final.csv",lineterminator="\n")
    print(test_data_new.shape)
    print(test_data_new['label'].unique())
        
    texts = []
    labels = []
    if BASE_TYPE == "HOPN":
            texts.extend(list(train_data_old['text']))
            texts.extend(list(val_data_old['text']))
            labels.extend(list(train_data_old['label']))
            labels.extend(list(val_data_old['label']))
            labels = [HOPN_label_map[i] for i in labels]
            
    else:   
        texts.extend(list(train_data_old['clean_text']))
        texts.extend(list(val_data_old['clean_text']))
        labels.extend(list(train_data_old['label']))
        labels.extend(list(val_data_old['label']))
#     if NEW_TYPE == "HOPN":
#         texts.extend(list(train_data_new['text']))
#         texts.extend(list(val_data_new['text']))
#     else:
#         texts.extend(list(train_data_new['clean_text']))
#         texts.extend(list(val_data_new['clean_text']))
#     if BASE_TYPE 
    

    
    c = list(zip(texts, labels))
    random.shuffle(c)
    texts, labels = zip(*c)
    train_texts = list(texts)
    train_labels = list(labels)
    assert len(train_texts)==len(train_labels)
    print("LEN TRAIN ", len(train_texts))
    
    texts = []
    labels = []
#     if BASE_TYPE=="HOPN":
#         texts.extend(list(test_data_old['text']))
#     else:
#         texts.extend(list(test_data_old['clean_text']))
    if NEW_TYPE=="HOPN":
        texts.extend(list(test_data_new['text']))
        labels.extend(list(test_data_new['label']))
        labels = [HOPN_label_map[i] for i in labels]
    else:
        texts.extend(list(test_data_new['clean_text']))
        labels.extend(list(test_data_new['label']))
    
    
    c = list(zip(texts, labels))
    random.shuffle(c)
    texts, labels = zip(*c)
    test_texts = list(texts)
    test_labels = list(labels)
    assert len(test_texts)==len(test_labels)
    print("LEN TEST ", len(test_texts))
    print(np.unique(np.asarray(labels)))
#     sys.exit(1)
    
    return train_texts, train_labels, test_texts, test_labels
    
    
    
if __name__=="__main__":
    stopwords2 = ['तुम','मेरी','मुझे','क्योंकि','हम','प्रति','अबकी','दे',
                              'आगे','', 'अर्थात', 'कुछ', 'तेरी', 'साबुत', 'अपनि', 'हूं',
                              'काफि', 'यिह', 'जा' ,'दे', 'देकर' ,'रह', 'कह' , 'कर' ,
                              'कहा', 'बात' , 'जिन्हों', 'किर', 'कोई','माननीय','शहर','बताएं',
                              'कौनसी','क्लिक','किसकी','बड़े','मैं','and','रही','आज','लें',
                              'आपके','मिलकर','सब','मेरे','जी','श्री','वैसा','आपका','अंदर',
                              'अत', 'अपना', 'अपनी', 'अपने', 'अभी', 'आदि', 'आप', 'इत्यादि',
                              'इन', 'इनका', 'इन्हीं', 'इन्हें', 'इन्हों', 'इस', 'इसका', 'इसकी', 
                              'इसके', 'इसमें', 'इसी', 'इसे', 'उन', 'उनका', 'उनकी', 'उनके',
                              'उनको', 'उन्हीं', 'उन्हें', 'उन्हों', 'उस', 'उसके', 'उसी', 'उसे',
                              'एक', 'एवं', 'एस', 'ऐसे', 'और', 'कई', 'कर','करता', 'करते',
                              'करना', 'करने', 'करें', 'कहते', 'कहा', 'का', 'काफ़ी', 'कि',
                              'कितना', 'किन्हें', 'किन्हों', 'किया', 'किर', 'किस', 'किसी',
                              'किसे', 'की', 'कुछ', 'कुल', 'के', 'को', 'कोई', 'कौन',
                              'कौनसा', 'गया', 'घर', 'जब', 'जहाँ', 'जा', 'जितना', 'जिन',
                              'जिन्हें', 'जिन्हों', 'जिस', 'जिसे', 'जीधर', 'जैसा', 'जैसे', 'जो',
                              'तक', 'तब', 'तरह', 'तिन', 'तिन्हें', 'तिन्हों', 'तिस',
                              'तिसे', 'तो', 'था', 'थी', 'थे', 'दबारा', 'दिया', 'दुसरा', 
                              'दूसरे', 'दो', 'द्वारा', 'न', 'नहीं', 'ना', 'निहायत', 'नीचे',
                              'ने', 'पर', 'पर', 'पहले', 'पूरा', 'पे', 'फिर', 'बनी', 'बही',
                              'बहुत', 'बाद', 'बाला', 'बिलकुल', 'भी', 'भीतर', 'मगर', 'मानो',
                              'मे', 'में', 'यदि', 'यह', 'यहाँ', 'यही', 'या', 'यिह', 'ये', 'रखें',
                              'रहा', 'रहे', 'ऱ्वासा', 'लिए', 'लिये', 'लेकिन', 'व', 'वर्ग', 'वह',
                              'वह', 'वहाँ', 'वहीं', 'वाले', 'वुह', 'वे', 'वग़ैरह', 'संग', 'सकता',
                              'सकते', 'सबसे', 'सभी', 'साथ', 'साबुत', 'साभ', 'सारा', 'से', 'सो',
                              'ही', 'हुआ', 'हुई', 'हुए', 'है', 'हैं', 'हो', 'होता', 'होती', 'होते',
                              'होना', 'होने', 'अपनि', 'जेसे', 'होति', 'सभि', 'तिंहों', 'इंहों', 'दवारा',
                              'इसि', 'किंहें', 'थि', 'उंहों', 'ओर', 'जिंहें', 'वहिं', 'अभि', 'बनि', 'हि',
                              'उंहिं', 'उंहें', 'हें', 'वगेरह', 'एसे', 'रवासा', 'कोन', 'निचे', 'काफि',
                              'उसि', 'पुरा', 'भितर', 'हे', 'बहि', 'वहां', 'कोइ', 'यहां', 'जिंहों', 
                              'तिंहें', 'किसि', 'कइ', 'यहि', 'इंहिं', 'जिधर', 'इंहें', 'अदि', 'इतयादि',
                              'हुइ', 'कोनसा', 'इसकि', 'दुसरे', 'जहां', 'अप', 'किंहों', 'उनकि', 'भि', 
                              'वरग', 'हुअ', 'जेसा', 'नहिं']

    start = datetime.now()
    print("\n Classification started started at ", start, "\n\n")
    
    set_random_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="HOPN")
    parser.add_argument("--new", default="HASOC19")
#     parser.add_argument("--ng",def)
    args = parser.parse_args()
    
    BASE_TYPE = args.base
    NEW_TYPE = args.new
    
    STOP_LIST = set(stopwords.words('english'))
    with open('mallet_stoplist.txt',"r") as f:         
        MALLET_STOPLIST = f.readlines()
    MALLET_STOPLIST = [x.replace("\n", "") for x in MALLET_STOPLIST] 
    STOP_LIST = STOP_LIST.union(set(MALLET_STOPLIST))
    STOP_LIST = list(STOP_LIST)
    STOP_LIST.extend(STOP_WORDS_HI)
    STOP_LIST.extend(STOP_WORDS_DE)
    STOP_LIST.extend(STOP_WORDS_ES)
    STOP_LIST.extend(["rt",":","…","&amp","","https://t.co/…","#ff", "ff"])
    STOP_LIST += stopwords2
    STOP_LIST = list(set(STOP_LIST))
    stemmer = PorterStemmer()
    table = str.maketrans(dict.fromkeys(string.punctuation))
    
    vectorizer_david = TfidfVectorizer(
                                   tokenizer=tokenize,
                                   preprocessor=preprocess,
                                   ngram_range=(1, 3),
                                   stop_words=STOP_LIST,
                                   use_idf=True,
                                   smooth_idf=False,
                                   norm=None,
                                   decode_error='replace',
                                   max_features=10000,
                                   min_df=5,
                                   max_df=0.75)
    
    print("\n\n---------------------")
    train_texts, train_labels, test_texts,test_labels = get_data()
    
    class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(train_labels),y=train_labels)
    class_weights = dict(zip(np.unique(train_labels), class_weights))
    print("Class weights:: ", class_weights)
    
    model = run_model(train_texts, train_labels,class_weights=class_weights)
    print("\n TEST ACCS")
    test_texts_feat = return_feature_set(test_texts)
    y_pred = model.predict(test_texts_feat)
    print("\n CLASSIFICATION REPORT \n", classification_report(test_labels, y_pred))
    print("\n CONFUSION MATRIX \n", confusion_matrix(test_labels, y_pred))
    print("\n ROC-AUC SCORE \n", roc_auc_score(test_labels,y_pred))
    print("\n MATH COEFF. \n", matthews_corrcoef(test_labels,y_pred))
    end = datetime.now()
    print("\n\nPredictions ended at ", end)
    print("\n\nTotal time taken ", end-start)
