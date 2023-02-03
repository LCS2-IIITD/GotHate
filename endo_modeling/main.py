# main driver code for HOPN

import argparse
import os
import numpy as np
import pandas as pd
import warnings
import gc
import random
import re
import pickle
import networkx as nx
from datetime import datetime
import gzip

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset, 
    WeightedRandomSampler,
    DataLoader
)

import transformers
from transformers import (
    BertTokenizerFast
)

import dgl
from dgl.dataloading import GraphDataLoader

print('Pytorch and CUDA Version: ', torch.__version__)
print('DGL Version: ', dgl.__version__)
print('Transformers Version: ', transformers.__version__)


from modeling_transformers.modeling_bert_new import (
    BertHateClassifier,
    HateBertModel
)

from utils.data_utils import get_graph_data
from utils.train_utils import (
    get_class_weights,
    train_fusion,
    train_dml
)

from modules.classifier import ClassifierHead

from config.config import *

SEED = 42
set_random_seed(SEED)

# -------------------------------------------------------------- DATA UTILS -------------------------------------------------------------- 

class HateDataset(Dataset):
    
    def __init__(
        self,
        data_type,
        path,
        exemplar_path,
        prompt_data_path,
        prompt_embedding_path
    ):

        self.data_type = data_type
        self.df = pd.read_csv(path)
        
        if USE_EXEMPLAR:
            with gzip.open(exemplar_path, 'rb') as file:
                self.exemplar_data = pickle.load(file)
            file.close()
        
        if USE_EVIDENCE_EARLY:
            print("\nUSE_EVIDENCE_EARLY\n")
            with gzip.open(prompt_data_path, 'rb') as file:
                self.prompt_data = pickle.load(file)
            file.close()

        elif USE_EVIDENCE:
            print("\nUSE_EVIDENCE\n")
            with gzip.open(prompt_embedding_path, 'rb') as file:
                self.prompt_embedding = pickle.load(file)
            file.close()
        
        print("\nLabel Distribution: \n", self.df['label'].value_counts())
        print("\n\nTopic Distribution: \n", self.df['topic'].value_counts(), "\n\n------------------------------------------------\n\n")         
                
        
        
    def __len__(
        self
    ):
        return len(self.df)
    
    
    
    
    def __getitem__(
        self,
        index: int
    ):
        if USE_EVIDENCE_EARLY and self.data_type != 'train':
            evidence = " [EVIDENCE] : " + self.prompt_data[self.df.iloc[index]['id']]
            text_inputs = TOKENIZER.encode_plus(
                str(self.df.iloc[index][COLUMN]),
                evidence,    
                max_length=TWEET_MAX_LEN,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_token_type_ids=True,            
                return_tensors='pt'
            )
        
        elif USE_FAQ and self.data_type != 'train':
            evidence = " [EVIDENCE] : " + TOPIC_INFO_DICT[str(self.df.iloc[index]['topic'])]
            text_inputs = TOKENIZER.encode_plus(
                str(self.df.iloc[index][COLUMN]),
                evidence,    
                max_length=TWEET_MAX_LEN,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_token_type_ids=True,            
                return_tensors='pt'
            )
            
        else:
            text_inputs = TOKENIZER.encode_plus(
                str(self.df.iloc[index][COLUMN]),
                max_length=TWEET_MAX_LEN,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_token_type_ids=True,            
                return_tensors='pt'
            )
        
        dataset_output = {
            'input_ids': text_inputs['input_ids'].flatten(),
            'attention_mask': text_inputs['attention_mask'].flatten(),
            'token_type_ids': text_inputs["token_type_ids"].flatten(),
            'targets': torch.tensor(self.df.iloc[index][LABEL], dtype=torch.long)
        }

        if USE_EXEMPLAR:
            dataset_output['exemplar_input'] = torch.tensor(self.exemplar_data[int(self.df.iloc[index]['id'])], dtype=torch.float)
        
        if USE_TIMELINE:
            dataset_output['timeline_input'] = torch.tensor(TIMELINE_DATA[int(self.df.iloc[index]['id'])], dtype=torch.float)
            
        if USE_GRAPH:
            dataset_output['graph_input'] = GRAPH_DATA[int(self.df['uid'].iloc[index])]
                
        if USE_EVIDENCE:
            tweet_id = self.df.iloc[index]['id']
            if tweet_id in self.prompt_embedding:
                dataset_output['evidence_input'] = torch.tensor(self.prompt_embedding[tweet_id], dtype=torch.float)
            else:
                dataset_output['evidence_input'] = torch.tensor(self.prompt_embedding[-999], dtype=torch.float)    
        
        return dataset_output
        
# ------------------------------------------------------------ MAIN MODEL ------------------------------------------------------------ #

if __name__ == "__main__":
    
    print('\n\nProgram started at ', datetime.now())
    print("\nProcess ID: ", os.getpid())
    print('\nUsing Column: ', COLUMN)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='mbert')
    parser.add_argument("--graph_size",type=int, default=100)
    parser.add_argument("--graph_embedding",type=str, default='node2vec')
    parser.add_argument("--training_type",type=str, default='fusion')
    parser.add_argument("--weighted_sampling",type=str, default='no')
    args = parser.parse_args()
    
    NUM_LABELS = 4
    print("\nNUM_LABELS: ", NUM_LABELS)
    
    # ---------------------------------------------------- model selection ----------------------------------------------------
    
    if USE_CUSTOM_MODEL:
        MODEL_NAME = CUSTOM_MODEL
        print("\nUsing USE_CUSTOM_MODEL: ", MODEL_NAME)
        modelparams = 177856516
        TIMELINE_PATH = 'timeline_embeddings_10_mBERT_mean.gz'
        EXEMPLAR_PATHS = [
            'train_semantic_search_stacked_exemplar_5_mBERT_300.gz',
            'val_semantic_search_stacked_exemplar_5_mBERT_300.gz',
            'test_semantic_search_stacked_exemplar_5_mBERT_300.gz'
        ]
        PROMPT_EMBEDDING_FILES = [
            'train_PROMPT_mbert_cls.gz',
            'val_PROMPT_mbert_cls.gz',
            'test_PROMPT_mbert_cls.gz',
        ]
        

    elif args.model_name == 'mbert':
        MODEL_NAME = 'bert-base-multilingual-cased'
        modelparams = 177856516
        TIMELINE_PATH = 'timeline_embeddings_10_mBERT_mean.gz'
        EXEMPLAR_PATHS = [
            'train_semantic_search_stacked_exemplar_5_mBERT_300.gz',
            'val_semantic_search_stacked_exemplar_5_mBERT_300.gz',
            'test_semantic_search_stacked_exemplar_5_mBERT_300.gz'
        ]
        PROMPT_EMBEDDING_FILES = [
            'train_PROMPT_mbert_cls.gz',
            'val_PROMPT_mbert_cls.gz',
            'test_PROMPT_mbert_cls.gz',
        ]
               
     
    
    # ---------------------------------------------------- MODEL INITIALIATION ----------------------------------------------------    
                
    print("\n\nLoading Bert: ", MODEL_NAME)

    if MODEL_TYPE == 'fusion':
        MODEL = BertHateClassifier.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
        print("Model loaded...\n")
        print("\n\nmodelparams: ", modelparams, "\n\n")
        MODEL.to(DEVICE)

    elif MODEL_TYPE == 'dml':
        MODEL = HateBertModel.from_pretrained(MODEL_NAME)
        print("Base Model loaded...\n")
        print("\n\nmodelparams: ", modelparams, "\n\n")
        MODEL.to(DEVICE)

        # 2. Exemplar Classifier
        EXEMPLAR_CLASSIFIER = ClassifierHead(model_dim=MODEL.config.hidden_size, num_labels=NUM_LABELS)
        print("\n\n EXEMPLAR_CLASSIFIER Total parameters: ", sum(p.numel() for p in EXEMPLAR_CLASSIFIER.parameters()))
        EXEMPLAR_CLASSIFIER.to(DEVICE)

        # 3. Timeline Classifier
        TIMELINE_CLASSIFIER = ClassifierHead(model_dim=MODEL.config.hidden_size, num_labels=NUM_LABELS)
        print("\n\n TIMELINE_CLASSIFIER Total parameters: ", sum(p.numel() for p in TIMELINE_CLASSIFIER.parameters()))
        TIMELINE_CLASSIFIER.to(DEVICE)

        if USE_GRAPH:
            GRAPH_CLASSIFIER = ClassifierHead(model_dim=MODEL.config.hidden_size, num_labels=NUM_LABELS)
            print("\n\n GRAPH_CLASSIFIER Total parameters: ", sum(p.numel() for p in TIMELINE_CLASSIFIER.parameters()))
            GRAPH_CLASSIFIER.to(DEVICE)

        elif USE_EVIDENCE:
            EVIDENCE_CLASSIFIER = ClassifierHead(model_dim=MODEL.config.hidden_size, num_labels=NUM_LABELS)
            print("\n\n EVIDENCE_CLASSIFIER Total parameters: ", sum(p.numel() for p in TIMELINE_CLASSIFIER.parameters()))
            EVIDENCE_CLASSIFIER.to(DEVICE)

    if USE_EVIDENCE_EARLY:
        TOKENIZER = BertTokenizerFast.from_pretrained(MODEL_NAME, additional_special_tokens=["[EVIDENCE]"])
        MODEL.resize_token_embeddings(len(TOKENIZER))
    else: 
        TOKENIZER = BertTokenizerFast.from_pretrained(MODEL_NAME)
        print("\n\nTokenizer loaded...\n")
      
    

    print("\nTrainable Parameters: \n")
    for name, param in MODEL.named_parameters():
        if param.requires_grad:
            print(name)


    pytorch_total_params = sum(p.numel() for p in MODEL.parameters())
    print("\n\nTotal parameters: ", pytorch_total_params)
    
    pytorch_total_train_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    print("Total trainable parameters: {}, Percentage: {}".format(pytorch_total_train_params, pytorch_total_train_params*100/pytorch_total_params))
    
    if MODEL_TYPE == 'dml':
        exemplar_params = sum(p.numel() for p in EXEMPLAR_CLASSIFIER.parameters() if p.requires_grad)
        timeline_params = sum(p.numel() for p in TIMELINE_CLASSIFIER.parameters() if p.requires_grad)
        
        if USE_GRAPH:
            graph_params = sum(p.numel() for p in GRAPH_CLASSIFIER.parameters() if p.requires_grad)
            extra = exemplar_params + timeline_params + graph_params + pytorch_total_params - modelparams
        elif USE_EVIDENCE:
            evidence_params = sum(p.numel() for p in EVIDENCE_CLASSIFIER.parameters() if p.requires_grad)
            extra = exemplar_params + timeline_params + evidence_params + pytorch_total_params - modelparams
        else:
            extra = exemplar_params + timeline_params + pytorch_total_params - modelparams
        
    else:
        extra = pytorch_total_params-modelparams
        
    print("Extra trainable parameters added:", extra)
    print("Percentage parameter change: ", extra*100/modelparams)
    
    
    # -------------------------------------------------- HELPER DATA LOADING --------------------------------------------------
    
    print("\nTIMELINE_PATH: ", TIMELINE_PATH)
    print("\nEXEMPLAR_PATHS: ", EXEMPLAR_PATHS)
    print("\nEXEMPLAR_EMBEDDING_PATH: ", EXEMPLAR_EMBEDDING_PATH)
    print("\nPROMPT_FILES: ", PROMPT_FILES)
    print("\nPROMPT_EMBEDDING_FILES: ", PROMPT_EMBEDDING_FILES)
    print("\nPROMPT_EMBEDDING_PATH: ", PROMPT_EMBEDDING_PATH)
    
    with gzip.open(TIMELINE_EMBEDDING_PATH+TIMELINE_PATH, 'rb') as file:
        TIMELINE_DATA = pickle.load(file)
    file.close()
    print("\n\nTimeline embedding size: ", len(TIMELINE_DATA))
    
    
    if USE_GRAPH:
        start = datetime.now()

        GRAPH_DATA_PATH = ''
        print("\nGRAPH_DATA_PATH: ", GRAPH_DATA_PATH)

        NODE_EMBEDDING_PATH = ''
        print("\nNODE_EMBEDDING_PATH: ", NODE_EMBEDDING_PATH)

        GRAPH_DATA = get_graph_data(graph_path=GRAPH_DATA_PATH, graph_embedding_path=NODE_EMBEDDING_PATH)
        print("\nGRAPH_DATA size: ", len(GRAPH_DATA))

        end = datetime.now()
        print('Time taken for graph loading: ', end-start)
    
    # -------------------------------------------------- ACTUAL DATA LOADING --------------------------------------------------
    
    # If WeightedRandomSampler 
    if args.weighted_sampling == "yes":
        CLASS_WEIGHTS = get_class_weights()
        print("\n\nCLASS_WEIGHTS size: ", len(CLASS_WEIGHTS))

        weighted_sampler = WeightedRandomSampler(
            weights=CLASS_WEIGHTS,
            num_samples=len(CLASS_WEIGHTS),
            replacement=True
        ) 
        
        train_data = HateDataset(
            data_type='train',
            path=INPUT_PATH+'train_final.csv',
            exemplar_path=EXEMPLAR_EMBEDDING_PATH+EXEMPLAR_PATHS[0],
            prompt_data_path=PROMPT_FILES[0],
            prompt_embedding_path=PROMPT_EMBEDDING_PATH+PROMPT_EMBEDDING_FILES[0]
        )
        if USE_GRAPH:
            train_dataloader = GraphDataLoader(
                train_data,
                batch_size=BATCH_SIZE,
                shuffle=False
            )
        else:
            train_dataloader = DataLoader(
                train_data,
                batch_size=BATCH_SIZE,
                sampler=weighted_sampler,
                shuffle=False
            )
            print("\n\nTraining data loaded, length:", len(train_data))
        
    else:    
        train_data = HateDataset(
            data_type='train',
            path=INPUT_PATH+'train_final.csv',
            exemplar_path=EXEMPLAR_EMBEDDING_PATH+EXEMPLAR_PATHS[0],
            prompt_data_path=PROMPT_FILES[0],
            prompt_embedding_path=PROMPT_EMBEDDING_PATH+PROMPT_EMBEDDING_FILES[0]
        )
        if USE_GRAPH:
            train_dataloader = GraphDataLoader(
                train_data,
                batch_size=BATCH_SIZE,
                shuffle=True
            )
        else:
            train_dataloader = DataLoader(
                train_data,
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            print("\n\nTraining data loaded, length:", len(train_data))


    val_data = HateDataset(
        data_type='val',
        path=INPUT_PATH+'val_final.csv',
        exemplar_path=EXEMPLAR_EMBEDDING_PATH+EXEMPLAR_PATHS[1],
        prompt_data_path=PROMPT_FILES[1],
        prompt_embedding_path=PROMPT_EMBEDDING_PATH+PROMPT_EMBEDDING_FILES[1]
    )
    if USE_GRAPH:
        val_dataloader = GraphDataLoader(
            val_data,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    else:
        val_dataloader = DataLoader(
            val_data,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    print("\n\nValidation data loaded, length:", len(val_data))


    test_data = HateDataset(
        data_type='test',
        path=INPUT_PATH+'test_final.csv',
        exemplar_path=EXEMPLAR_EMBEDDING_PATH+EXEMPLAR_PATHS[2],
        prompt_data_path=PROMPT_FILES[2],
        prompt_embedding_path=PROMPT_EMBEDDING_PATH+PROMPT_EMBEDDING_FILES[2]
    )
    if USE_GRAPH:
        test_dataloader = GraphDataLoader(
            test_data,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    else:
        test_dataloader = DataLoader(
            test_data,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    print("\n\nTest data loaded, length:", len(test_data))
    
    del train_data
    del val_data
    del test_data
    gc.collect()
    
    
    
    # ------------------------------ TRAINING SETUP ------------------------------ #
    
    start = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print("\n\nTraining started started at ", start)
    
    model_info = {
        'model_name': args.model_name,
        'timestamp': start
    }
    
    if MODEL_TYPE == 'fusion':
        train_fusion(model_info=model_info,
                     model=MODEL,
                     train_data_loader=train_dataloader,
                     val_data_loader=val_dataloader,
                     test_data_loader=test_dataloader,
                     learning_rate=BASE_LEARNING_RATE, 
                     weight_decay=WEIGHT_DECAY,
                     num_labels=NUM_LABELS)
    
    if MODEL_TYPE == 'dml':
        if USE_GRAPH:
            models=[MODEL, EXEMPLAR_CLASSIFIER, TIMELINE_CLASSIFIER, GRAPH_CLASSIFIER]
        elif USE_EVIDENCE:
            models=[MODEL, EXEMPLAR_CLASSIFIER, TIMELINE_CLASSIFIER, EVIDENCE_CLASSIFIER]
        else:
            models=[MODEL, EXEMPLAR_CLASSIFIER, TIMELINE_CLASSIFIER]
        train_dml(model_info=model_info,
                  models=models,
                  train_data_loader=train_dataloader,
                  val_data_loader=val_dataloader,
                  test_data_loader=test_dataloader,
                  base_learning_rate=BASE_LEARNING_RATE, 
                  classifier_learning_rate=CLASSIFIER_LEARNING_RATE, 
                  weight_decay=WEIGHT_DECAY,
                  num_labels=NUM_LABELS)
    end = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print("\n\nTraining ended ended at ", end)
    print("\n\nTotal time taken ", end-start)