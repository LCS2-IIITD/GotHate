# Code for training utils for HOPN

import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW  

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score, 
    recall_score, 
    precision_score, 
    accuracy_score,
    classification_report,
    confusion_matrix
)

from config.config import *
from .losses import *



def get_class_weights():
    df = pd.read_csv(INPUT_PATH+'train_final.csv')
    target_list = torch.tensor(df.label)
    class_count = np.bincount(df.label)
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    class_weights_all = class_weights[target_list]
    
    del df
    del target_list
    del class_count
    del class_weights
    
    return class_weights_all



def get_prediction_scores(
    preds,
    gold
):
    return {
        'accuracy': accuracy_score(gold, preds),
        'precision': precision_score(gold, preds, average='macro'),
        'recall': recall_score(gold, preds, average='macro'),
        'f1-score': f1_score(gold, preds, average='macro'),
    }



# ---------------------------------------------- Train utils for Fusion HOPN ---------------------------------------------- #

def train_epoch_fusion(
    model,
    data_loader,
    optimizer
):
    model.train()
    epoch_train_loss = 0.0
    pred_list = []
    gold_list = []
    for step, batch in enumerate(tqdm(data_loader, desc="Training Iteration")):

        optimizer.zero_grad()
        
        model_kwargs = {
            'input_ids' : batch['input_ids'].to(DEVICE, dtype = torch.long),
            'attention_mask': batch['attention_mask'].to(DEVICE, dtype = torch.long),
            'token_type_ids': batch['token_type_ids'].to(DEVICE, dtype = torch.long),
            'labels': batch['targets'].to(DEVICE, dtype = torch.long)
        }  
        
        if USE_EXEMPLAR:
                model_kwargs['exemplar_input'] = batch['exemplar_input'].to(DEVICE)
                
        if USE_TIMELINE:
                model_kwargs['timeline_input'] = batch['timeline_input'].to(DEVICE)
        
        if USE_GRAPH:
                model_kwargs['graph_input'] = batch['graph_input'].to(DEVICE)
                
        if USE_EVIDENCE:
            model_kwargs['evidence_input'] = batch['evidence_input'].to(DEVICE)
            
        outputs = model(**model_kwargs)
        loss = outputs['loss']
        epoch_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs['logits'], axis=1).detach().cpu().numpy().tolist()
        gold = model_kwargs['labels'].cpu().tolist()

        pred_list.extend(preds)
        gold_list.extend(gold)
    
    return round(epoch_train_loss/ step, 4), pred_list, gold_list



def val_epoch_fusion(
    model,
    data_loader
):
    model.eval()
    
    epoch_val_loss = 0.0
    pred_list=[]
    gold_list = []
    for step, batch in enumerate(tqdm(data_loader, desc="Validation Iteration")):  
        
        with torch.no_grad():
            
            model_kwargs = {
                'input_ids' : batch['input_ids'].to(DEVICE, dtype = torch.long),
                'attention_mask': batch['attention_mask'].to(DEVICE, dtype = torch.long),
                'token_type_ids': batch['token_type_ids'].to(DEVICE, dtype = torch.long),
                'labels': batch['targets'].to(DEVICE, dtype = torch.long)
            }  

            if USE_EXEMPLAR:
                    model_kwargs['exemplar_input'] = batch['exemplar_input'].to(DEVICE)

            if USE_TIMELINE:
                    model_kwargs['timeline_input'] = batch['timeline_input'].to(DEVICE)

            if USE_GRAPH:
                    model_kwargs['graph_input'] = batch['graph_input'].to(DEVICE)

            if USE_EVIDENCE:
                model_kwargs['evidence_input'] = batch['evidence_input'].to(DEVICE)

            outputs = model(**model_kwargs)
            loss = outputs['loss']
            epoch_val_loss += loss.item()

            preds = torch.argmax(outputs['logits'], axis=1).detach().cpu().numpy().tolist()
            gold = model_kwargs['labels'].cpu().tolist()

            pred_list.extend(preds)
            gold_list.extend(gold)
        
    return round(epoch_val_loss/ step, 4), pred_list, gold_list



def test_epoch_fusion(
    model,
    data_loader
):
    model.eval()
    
    epoch_val_loss = 0.0
    pred_list=[]
    gold_list = []
    for step, batch in enumerate(tqdm(data_loader, desc="Test Iteration")):  
        
        with torch.no_grad():
            
            model_kwargs = {
                'input_ids' : batch['input_ids'].to(DEVICE, dtype = torch.long),
                'attention_mask': batch['attention_mask'].to(DEVICE, dtype = torch.long),
                'token_type_ids': batch['token_type_ids'].to(DEVICE, dtype = torch.long),
            }
            
            labels = batch['targets'].to(DEVICE, dtype = torch.long)

            if USE_EXEMPLAR:
                    model_kwargs['exemplar_input'] = batch['exemplar_input'].to(DEVICE)

            if USE_TIMELINE:
                    model_kwargs['timeline_input'] = batch['timeline_input'].to(DEVICE)

            if USE_GRAPH:
                    model_kwargs['graph_input'] = batch['graph_input'].to(DEVICE)

            if USE_EVIDENCE:
                model_kwargs['evidence_input'] = batch['evidence_input'].to(DEVICE)

            outputs = model(**model_kwargs)
            
            preds = torch.argmax(outputs['logits'], axis=1).detach().cpu().numpy().tolist()
            gold = labels.cpu().tolist()

            pred_list.extend(preds)
            gold_list.extend(gold)

    return pred_list, gold_list





def prepare_for_training_fusion(
    model,
    learning_rate: float,
    weight_decay: float
):
    optimizer=AdamW(model.parameters(), 
                    lr=learning_rate, 
                    weight_decay=weight_decay)
    gc.collect()
    return optimizer



def train_fusion(
    model_info,
    model,
    train_data_loader,
    val_data_loader,
    test_data_loader,
    learning_rate,
    weight_decay,
    num_labels
):
    
    optimizer = prepare_for_training_fusion(model=model,
                                            learning_rate=learning_rate, 
                                            weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    patience = 0
    
    for epoch in range(MAX_EPOCHS):
        
        # Train Set
        train_loss, train_preds, train_gold = train_epoch_fusion(model,
                                                                 train_data_loader, 
                                                                 optimizer)
        train_losses.append(train_loss)
        train_results = get_prediction_scores(train_preds, train_gold)
        train_cr = classification_report(y_true=train_gold, y_pred=train_preds, output_dict=True)
        train_cm = confusion_matrix(y_true=train_gold, y_pred=train_preds)
        
        # Val Set
        val_loss, val_preds, val_gold = val_epoch_fusion(model,
                                                         val_data_loader)
        val_losses.append(val_loss)
        val_results = get_prediction_scores(val_preds, val_gold)
        val_cr = classification_report(y_true=val_gold, y_pred=val_preds, output_dict=True)
        val_cm = confusion_matrix(y_true=val_gold, y_pred=val_preds)
        
        # Test Set
        test_preds, test_gold = test_epoch_fusion(model,
                                                  test_data_loader)
        test_results = get_prediction_scores(test_preds, test_gold)
        test_cr = classification_report(y_true=test_gold, y_pred=test_preds, output_dict=True)
        test_cm = confusion_matrix(y_true=test_gold, y_pred=test_preds)
        
        print("\nEpoch: {}\ttrain_loss: {}\tval_loss: {}\tmin_val_loss: {}".format(epoch+1, 
                                                                                   train_loss, 
                                                                                   val_loss, 
                                                                                   min(val_losses)))
        
        # ----------------------------------------------------------- Train Results --------------------------------------------------------
       
        print("\ntrain_acc: {}\ttrain_precision: {}\ttrain_recall: {}\ttrain_f1: {}".format(train_results['accuracy'], 
                                                                                            train_results['precision'], 
                                                                                            train_results['recall'], 
                                                                                            train_results['f1-score']))
        
        print("\ntrain_hate_precision: {}\ttrain_hate_recall: {}\ttrain_hate_f1: {}".format(train_cr['0']['precision'], 
                                                                                            train_cr['0']['recall'],
                                                                                            train_cr['0']['f1-score']))

        print("\ntrain_offensive_precision: {}\ttrain_offensive_recall: {}\ttrain_offensive_f1: {}".format(train_cr['1']['precision'], 
                                                                                                           train_cr['1']['recall'],
                                                                                                           train_cr['1']['f1-score']))

        print("\ntrain_provocative_precision: {}\ttrain_provocative_recall: {}\ttrain_provocative_f1: {}".format(train_cr['2']['precision'], 
                                                                                                                 train_cr['2']['recall'],
                                                                                                                 train_cr['2']['f1-score']))

        print("\ntrain_control_precision: {}\ttrain_control_recall: {}\ttrain_control_f1: {}".format(train_cr['3']['precision'], 
                                                                                                     train_cr['3']['recall'],
                                                                                                     train_cr['3']['f1-score']))
        print("\nTrain confusion matrix: ", train_cm)

        # ----------------------------------------------------------- Val Results --------------------------------------------------------
        
        print("\n\nval_acc: {}\tval_precision: {}\tval_recall: {}\tval_f1: {}".format(val_results['accuracy'], 
                                                                                      val_results['precision'], 
                                                                                      val_results['recall'], 
                                                                                      val_results['f1-score']))

        print("\nval_hate_precision: {}\tval_hate_recall: {}\tval_hate_f1: {}".format(val_cr['0']['precision'], 
                                                                                      val_cr['0']['recall'],
                                                                                      val_cr['0']['f1-score']))

        print("\nval_offensive_precision: {}\tval_offensive_recall: {}\tval_offensive_f1: {}".format(val_cr['1']['precision'], 
                                                                                                     val_cr['1']['recall'],
                                                                                                     val_cr['1']['f1-score']))

        print("\nval_provocative_precision: {}\tval_provocative_recall: {}\tval_provocative_f1: {}".format(val_cr['2']['precision'], 
                                                                                                           val_cr['2']['recall'],
                                                                                                           val_cr['2']['f1-score']))

        print("\nval_control_precision: {}\tval_control_recall: {}\tval_control_f1: {}".format(val_cr['3']['precision'], 
                                                                                               val_cr['3']['recall'],
                                                                                               val_cr['3']['f1-score']))
        print("\nVal confusion matrix: ", val_cm)

        # ----------------------------------------------------------- Test Results --------------------------------------------------------
        
        print("\n\ntest_acc: {}\ttest_precision: {}\ttest_recall: {}\ttest_f1: {}".format(test_results['accuracy'], 
                                                                                          test_results['precision'], 
                                                                                          test_results['recall'], 
                                                                                          test_results['f1-score']))
        
        print("\ntest_hate_precision: {}\ttest_hate_recall: {}\ttest_hate_f1: {}".format(test_cr['0']['precision'], 
                                                                                         test_cr['0']['recall'],
                                                                                         test_cr['0']['f1-score']))

        print("\ntest_offensive_precision: {}\ttest_offensive_recall: {}\ttest_offensive_f1: {}".format(test_cr['1']['precision'], 
                                                                                                        test_cr['1']['recall'],
                                                                                                        test_cr['1']['f1-score']))

        print("\ntest_provocative_precision: {}\ttest_provocative_recall: {}\ttest_provocative_f1: {}".format(test_cr['2']['precision'], 
                                                                                                              test_cr['2']['recall'],
                                                                                                              test_cr['2']['f1-score']))

        print("\ntest_control_precision: {}\ttest_control_recall: {}\ttest_control_f1: {}".format(test_cr['3']['precision'], 
                                                                                                  test_cr['3']['recall'],
                                                                                                  test_cr['3']['f1-score']))
        print("\nTest confusion matrix: ", test_cm)

        # --------------------------------------------- Storing Val Results ---------------------------------------------
        
        if SAVE_RESULTS:
            val_result_df = pd.DataFrame(list(zip(val_gold, val_preds)), columns=['gold', 'preds'])

            val_folder = + MODEL_TYPE + '/' + str(model_info['model_name']) + '/val/' + str(model_info['timestamp']) + '/'

            if not os.path.exists(val_folder):
                print("\n\nCreating folder at: ", val_folder)
                os.makedirs(val_folder)

            val_path = val_folder + str(model_info['model_name']) + '_fusion_timestamp_' + str(model_info['timestamp']) +'_val_epoch_' + str(epoch+1) + '.csv'

            val_result_df.to_csv(val_path, index=False)
            print("\nStored val data at: ", val_path)

            # --------------------------------------------- Storing Test Results ---------------------------------------------

            test_result_df = pd.DataFrame(list(zip(test_gold, test_preds)), columns=['gold', 'preds'])

            test_folder = '' + MODEL_TYPE + '/' + str(model_info['model_name']) + '/test/' + str(model_info['timestamp']) + '/'

            if not os.path.exists(test_folder):
                print("\n\nCreating folder at: ", test_folder)
                os.makedirs(test_folder)

            test_path = test_folder + str(model_info['model_name']) + '_fusion_timestamp_' + str(model_info['timestamp']) +'_test_epoch_' + str(epoch+1) + '.csv'

            test_result_df.to_csv(test_path, index=False)
            print("\nStored test data at: ", test_path)

        
        del train_loss
        del val_loss
        del train_preds
        del train_gold
        del train_results
        del train_cr
        del val_preds
        del val_gold
        del val_results
        del val_cr
        del test_preds
        del test_gold
        del test_results
        del test_cr
        gc.collect()
        torch.cuda.empty_cache()

# --------------------------------------------------- Train utils for DML HOPN --------------------------------------------------- #

def prepare_for_training_dml(
    models,
    base_learning_rate: float,
    classifier_learning_rate: float,
    weight_decay: float
):
    if USE_EVIDENCE or USE_GRAPH:
        return [
            AdamW(models[0].parameters(), lr=base_learning_rate, weight_decay=weight_decay),
            AdamW(models[1].parameters(), lr=classifier_learning_rate, weight_decay=weight_decay),
            AdamW(models[2].parameters(), lr=classifier_learning_rate, weight_decay=weight_decay),
            AdamW(models[3].parameters(), lr=classifier_learning_rate, weight_decay=weight_decay)
        ]
    else:
        return [
            AdamW(models[0].parameters(), lr=base_learning_rate, weight_decay=weight_decay),
            AdamW(models[1].parameters(), lr=classifier_learning_rate, weight_decay=weight_decay),
            AdamW(models[2].parameters(), lr=classifier_learning_rate, weight_decay=weight_decay),
        ]
    


    
def get_dml_loss(
    logits1, 
    logits2, 
    true_labels
):
    classification_loss_fct = nn.CrossEntropyLoss().to(DEVICE)
#     classification_loss_fct = MultiFocalLoss(num_class=4).to(DEVICE)
    divergence_loss_fct = nn.KLDivLoss(reduction="batchmean").to(DEVICE)

    classification_loss = classification_loss_fct(logits1.view(-1, 4), true_labels.view(-1))
    kldiv_loss = divergence_loss_fct(F.log_softmax(logits1.view(-1, 4), dim=-1), F.softmax(logits2.view(-1, 4), dim=-1))
    return classification_loss + (KL_LOSS_LAMBDA*kldiv_loss)



def get_dml_loss_triplet(
    logits1, 
    logits2, 
    logits3,
    true_labels
):
    classification_loss_fct = nn.CrossEntropyLoss().to(DEVICE)
#     classification_loss_fct = MultiFocalLoss(num_class=4).to(DEVICE)
    divergence_loss_fct1 = nn.KLDivLoss(reduction="batchmean").to(DEVICE)
    divergence_loss_fct2 = nn.KLDivLoss(reduction="batchmean").to(DEVICE)

    classification_loss = classification_loss_fct(logits1.view(-1, 4), true_labels.view(-1))
    kldiv_loss1 = divergence_loss_fct1(F.log_softmax(logits1.view(-1, 4), dim=-1), F.softmax(logits2.view(-1, 4), dim=-1))
    kldiv_loss2 = divergence_loss_fct2(F.log_softmax(logits1.view(-1, 4), dim=-1), F.softmax(logits3.view(-1, 4), dim=-1))
    return classification_loss + (KL_LOSS_LAMBDA*((kldiv_loss1+kldiv_loss2)/2))



def get_preds(
    logits1,
    logits2,
    combination_type: str
):
    logits1 = F.softmax(logits1, dim=-1)
    logits2 = F.softmax(logits2, dim=-1)
    
    if combination_type == 'average':
        return torch.argmax((logits1 + logits2 )/2, axis=1).detach().cpu().numpy().tolist()
    
    elif combination_type == 'confidence':
        labels1 = torch.argmax(logits1, axis=1).detach().cpu().numpy().tolist()
        labels2 = torch.argmax(logits2, axis=1).detach().cpu().numpy().tolist()
        
        confidence1 = torch.max(logits1, dim=1)[0].tolist()
        confidence2 = torch.max(logits2, dim=1)[0].tolist()
        
        final_label = []
        for i in range(len(labels1)):
            confidences = torch.tensor([confidence1[i], confidence2[i]])
            if torch.max(confidences) == confidence1[i]:
                final_labels.append(labels1[i])

            else:
                final_labels.append(labels2[i])
          
        return final_label


    
def get_preds_triplet(
    logits1,
    logits2,
    logits3,
    combination_type: str
):
    logits1 = F.softmax(logits1, dim=-1)
    logits2 = F.softmax(logits2, dim=-1)
    logits3 = F.softmax(logits3, dim=-1)
    
    if combination_type == 'average':
        return torch.argmax((logits1 + logits2 + logits3)/3, axis=1).detach().cpu().numpy().tolist()
    
    elif combination_type == 'confidence':
        labels1 = torch.argmax(logits1, axis=1).detach().cpu().numpy().tolist()
        labels2 = torch.argmax(logits2, axis=1).detach().cpu().numpy().tolist()
        labels3 = torch.argmax(logits3, axis=1).detach().cpu().numpy().tolist()
        
        confidence1 = torch.max(logits1, dim=1)[0].tolist()
        confidence2 = torch.max(logits2, dim=1)[0].tolist()
        confidence3 = torch.max(logits3, dim=1)[0].tolist()
        
        final_label = []
        for i in range(len(labels1)):
            confidences = torch.tensor([confidence1[i], confidence2[i], confidence3[i]])
            if torch.max(confidences) == confidence1[i]:
                final_labels.append(labels1[i])

            elif torch.max(confidences) == confidence2[i]:
                final_labels.append(labels2[i])

            elif torch.max(confidences) == confidence3[i]:
                final_labels.append(labels3[i])
          
        return final_label
    
    
    
def train_epoch_dml(
    models,
    optimizers,
    data_loader
):
    if USE_GRAPH or USE_EVIDENCE:
        model, exemplar_classifier, timeline_classifier, extra_classifier = models
        base_optimizer, exemplar_optimizer, timeline_optimizer, extra_optimizer = optimizers
        extra_classifier.train()
        epoch_extra_loss = 0.0
        
    else:
        model, exemplar_classifier, timeline_classifier = models
        base_optimizer, exemplar_optimizer, timeline_optimizer = optimizers
    
    model.train()
    exemplar_classifier.train()
    timeline_classifier.train()
        
    epoch_exemplar_loss = 0.0
    epoch_timeline_loss = 0.0
    epoch_loss = 0.0    

    pred_list = []
    gold_list = []
    for step, batch in enumerate(tqdm(data_loader, desc="Training Iteration")):
        
        base_optimizer.zero_grad()
        exemplar_optimizer.zero_grad()
        timeline_optimizer.zero_grad()
        
        model_kwargs = {
            'input_ids' : batch['input_ids'].to(DEVICE, dtype = torch.long),
            'attention_mask': batch['attention_mask'].to(DEVICE, dtype = torch.long),
            'token_type_ids': batch['token_type_ids'].to(DEVICE, dtype = torch.long),
            'exemplar_input': batch['exemplar_input'].to(DEVICE),
            'timeline_input':  batch['timeline_input'].to(DEVICE),
        }  
        labels = batch['targets'].to(DEVICE, dtype = torch.long)
        
        # ----------------------------------------- Forward pass to base model -----------------------------------------
        
        if USE_GRAPH:
                extra_optimizer.zero_grad()
                model_kwargs['graph_input'] = batch['graph_input'].to(DEVICE)
                outputs = model(**model_kwargs)[1]
                extra_output = outputs[:, model.config.hidden_size*2:model.config.hidden_size*3]
                extra_logits = extra_classifier(input_tensor=extra_output)
                
        elif USE_EVIDENCE:
            extra_optimizer.zero_grad()
            model_kwargs['evidence_input'] = batch['evidence_input'].to(DEVICE)
            outputs = model(**model_kwargs)[1]
            extra_output = outputs[:, model.config.hidden_size*2:model.config.hidden_size*3]
            extra_logits = extra_classifier(input_tensor=extra_output)
            
        else:
            outputs = model(**model_kwargs)[1]
            
        
        # ----------------------------------------- Forward pass to classifiers -----------------------------------------
        
        exemplar_output = outputs[:, :model.config.hidden_size]
        timeline_output = outputs[:, model.config.hidden_size:model.config.hidden_size*2]
        
        exemplar_logits = exemplar_classifier(input_tensor=exemplar_output)
        timeline_logits = timeline_classifier(input_tensor=timeline_output)
        
        # ----------------------------------------- Computing loss and backward pass -----------------------------------------
     
        if USE_EVIDENCE or USE_GRAPH:
            
            # 2. Exemplar Loss
            exemplar_loss = get_dml_loss_triplet(
                logits1=exemplar_logits, 
                logits2=timeline_logits, 
                logits3=extra_logits, 
                true_labels=labels
            )
            epoch_exemplar_loss += exemplar_loss.item()

            # 3. Timeline Loss
            timeline_loss = get_dml_loss_triplet(
                logits1=timeline_logits, 
                logits2=exemplar_logits, 
                logits3=extra_logits, 
                true_labels=labels
            )
            epoch_timeline_loss += timeline_loss.item()
            
            # 3. Evidence Loss
            extra_loss = get_dml_loss_triplet(
                logits1=extra_logits, 
                logits2=exemplar_logits, 
                logits3=timeline_logits, 
                true_labels=labels
            )
            epoch_extra_loss += extra_loss.item()

            loss = (exemplar_loss + timeline_loss + epoch_extra_loss)/3
            epoch_loss += loss.item()

            loss.backward()

            base_optimizer.step()
            exemplar_optimizer.step()
            timeline_optimizer.step()
            extra_optimizer.step()
            
            # ----------------------------------------- Get labels -----------------------------------------

            preds = get_preds_triplet(
                logits1=exemplar_logits,
                logits2=timeline_logits,
                logits3=extra_logits,
                combination_type=LOGITS_COMBINATION_TYPE
            )
            gold = labels.cpu().tolist()

            pred_list.extend(preds)
            gold_list.extend(gold)
            
        else:
            # 2. Exemplar Loss
            exemplar_loss = get_dml_loss(
                logits1=exemplar_logits, 
                logits2=timeline_logits, 
                true_labels=labels
            )
            epoch_exemplar_loss += exemplar_loss.item()

            # 2. Exemplar Loss
            timeline_loss = get_dml_loss(
                logits1=timeline_logits, 
                logits2=exemplar_logits, 
                true_labels=labels
            )
            epoch_timeline_loss += timeline_loss.item()

            loss = (exemplar_loss + timeline_loss)/2
            epoch_loss += loss.item()

            loss.backward()

            base_optimizer.step()
            exemplar_optimizer.step()
            timeline_optimizer.step()

            # ----------------------------------------- Get labels -----------------------------------------

            preds = get_preds(
                logits1=exemplar_logits,
                logits2=timeline_logits,
                combination_type=LOGITS_COMBINATION_TYPE
            )
            gold = labels.cpu().tolist()

            pred_list.extend(preds)
            gold_list.extend(gold)
        
    if USE_EVIDENCE or USE_GRAPH:
        losses = {
            'exemplar_loss': round(epoch_exemplar_loss/ step, 4),
            'timeline_loss': round(epoch_timeline_loss/ step, 4),
            'extra_loss': round(epoch_extra_loss/ step, 4),
            'loss': round(epoch_loss/ step, 4)
        }

        
    else:    
        losses = {
            'exemplar_loss': round(epoch_exemplar_loss/ step, 4),
            'timeline_loss': round(epoch_timeline_loss/ step, 4),
            'loss': round(epoch_loss/ step, 4)
        }

    return losses, pred_list, gold_list




def val_epoch_dml(
    models,
    data_loader
):
    if USE_GRAPH or USE_EVIDENCE:
        model, exemplar_classifier, timeline_classifier, extra_classifier = models
        extra_classifier.eval()
        epoch_extra_loss = 0.0

    else:
        model, exemplar_classifier, timeline_classifier = models
    
    model.eval()
    exemplar_classifier.eval()
    timeline_classifier.eval()
        
    epoch_exemplar_loss = 0.0
    epoch_timeline_loss = 0.0
    epoch_loss = 0.0    
    

    pred_list = []
    gold_list = []
    for step, batch in enumerate(tqdm(data_loader, desc="Validation Iteration")):
        
        with torch.no_grad():
            
            model_kwargs = {
                'input_ids' : batch['input_ids'].to(DEVICE, dtype = torch.long),
                'attention_mask': batch['attention_mask'].to(DEVICE, dtype = torch.long),
                'token_type_ids': batch['token_type_ids'].to(DEVICE, dtype = torch.long),
                'exemplar_input': batch['exemplar_input'].to(DEVICE),
                'timeline_input':  batch['timeline_input'].to(DEVICE),
            }  
            labels = batch['targets'].to(DEVICE, dtype = torch.long)
            
            if USE_GRAPH:
                model_kwargs['graph_input'] = batch['graph_input'].to(DEVICE)
                outputs = model(**model_kwargs)[1]
                extra_output = outputs[:, model.config.hidden_size*2:model.config.hidden_size*3]
                extra_logits = extra_classifier(input_tensor=extra_output)
                
            elif USE_EVIDENCE:
                model_kwargs['evidence_input'] = batch['evidence_input'].to(DEVICE)
                outputs = model(**model_kwargs)[1]
                extra_output = outputs[:, model.config.hidden_size*2:model.config.hidden_size*3]
                extra_logits = extra_classifier(input_tensor=extra_output)

            else:
                outputs = model(**model_kwargs)[1]
                
            # ----------------------------------------- Forward pass to classifiers -----------------------------------------

            exemplar_output = outputs[:, :model.config.hidden_size]
            timeline_output = outputs[:, model.config.hidden_size:model.config.hidden_size*2]

            exemplar_logits = exemplar_classifier(input_tensor=exemplar_output)
            timeline_logits = timeline_classifier(input_tensor=timeline_output)

            # ----------------------------------------- Computing loss -----------------------------------------

            if USE_EVIDENCE or USE_GRAPH:
            
                # 2. Exemplar Loss
                exemplar_loss = get_dml_loss_triplet(
                    logits1=exemplar_logits, 
                    logits2=timeline_logits, 
                    logits3=extra_logits, 
                    true_labels=labels
                )
                epoch_exemplar_loss += exemplar_loss.item()

                # 3. Timeline Loss
                timeline_loss = get_dml_loss_triplet(
                    logits1=timeline_logits, 
                    logits2=exemplar_logits, 
                    logits3=extra_logits, 
                    true_labels=labels
                )
                epoch_timeline_loss += timeline_loss.item()

                # 3. Evidence Loss
                extra_loss = get_dml_loss_triplet(
                    logits1=extra_logits, 
                    logits2=exemplar_logits, 
                    logits3=timeline_logits, 
                    true_labels=labels
                )
                epoch_extra_loss += extra_loss.item()

                loss = (exemplar_loss + timeline_loss + epoch_extra_loss)/3
                epoch_loss += loss.item()

                # ----------------------------------------- Get labels -----------------------------------------

                preds = get_preds_triplet(
                    logits1=exemplar_logits,
                    logits2=timeline_logits,
                    logits3=extra_logits,
                    combination_type=LOGITS_COMBINATION_TYPE
                )
                gold = labels.cpu().tolist()

                pred_list.extend(preds)
                gold_list.extend(gold)

            else:
                # 2. Exemplar Loss
                exemplar_loss = get_dml_loss(
                    logits1=exemplar_logits, 
                    logits2=timeline_logits, 
                    true_labels=labels
                )
                epoch_exemplar_loss += exemplar_loss.item()

                # 2. Exemplar Loss
                timeline_loss = get_dml_loss(
                    logits1=timeline_logits, 
                    logits2=exemplar_logits, 
                    true_labels=labels
                )
                epoch_timeline_loss += timeline_loss.item()

                loss = (exemplar_loss + timeline_loss)/2
                epoch_loss += loss.item()

                # ----------------------------------------- Get labels -----------------------------------------

                preds = get_preds(
                    logits1=exemplar_logits,
                    logits2=timeline_logits,
                    combination_type=LOGITS_COMBINATION_TYPE
                )
                gold = labels.cpu().tolist()

                pred_list.extend(preds)
                gold_list.extend(gold)
        
    if USE_EVIDENCE or USE_GRAPH:
        losses = {
            'exemplar_loss': round(epoch_exemplar_loss/ step, 4),
            'timeline_loss': round(epoch_timeline_loss/ step, 4),
            'extra_loss': round(epoch_extra_loss/ step, 4),
            'loss': round(epoch_loss/ step, 4)
        }

        
    else:    
        losses = {
            'exemplar_loss': round(epoch_exemplar_loss/ step, 4),
            'timeline_loss': round(epoch_timeline_loss/ step, 4),
            'loss': round(epoch_loss/ step, 4)
        }
    
    return losses, pred_list, gold_list



def test_epoch_dml(
    models,
    data_loader
):
    if USE_GRAPH or USE_EVIDENCE:
        model, exemplar_classifier, timeline_classifier, extra_classifier = models
        extra_classifier.eval()

    else:
        model, exemplar_classifier, timeline_classifier = models
    
    model.eval()
    exemplar_classifier.eval()
    timeline_classifier.eval()
        
    pred_list = []
    gold_list = []
    for step, batch in enumerate(tqdm(data_loader, desc="Test Iteration")):
        
        with torch.no_grad():
            
            model_kwargs = {
                'input_ids' : batch['input_ids'].to(DEVICE, dtype = torch.long),
                'attention_mask': batch['attention_mask'].to(DEVICE, dtype = torch.long),
                'token_type_ids': batch['token_type_ids'].to(DEVICE, dtype = torch.long),
                'exemplar_input': batch['exemplar_input'].to(DEVICE),
                'timeline_input':  batch['timeline_input'].to(DEVICE),
            }  
            labels = batch['targets'].to(DEVICE, dtype = torch.long)
            
            if USE_GRAPH:
                model_kwargs['graph_input'] = batch['graph_input'].to(DEVICE)
                outputs = model(**model_kwargs)[1]
                extra_output = outputs[:, model.config.hidden_size*2:model.config.hidden_size*3]
                extra_logits = extra_classifier(input_tensor=extra_output)
                
            elif USE_EVIDENCE:
                model_kwargs['evidence_input'] = batch['evidence_input'].to(DEVICE)
                outputs = model(**model_kwargs)[1]
                extra_output = outputs[:, model.config.hidden_size*2:model.config.hidden_size*3]
                extra_logits = extra_classifier(input_tensor=extra_output)

            else:
                outputs = model(**model_kwargs)[1]

            # ----------------------------------------- Forward pass to classifiers -----------------------------------------

            exemplar_output = outputs[:, :model.config.hidden_size]
            timeline_output = outputs[:, model.config.hidden_size:model.config.hidden_size*2]

            exemplar_logits = exemplar_classifier(input_tensor=exemplar_output)
            timeline_logits = timeline_classifier(input_tensor=timeline_output)


            # ----------------------------------------- Get labels -----------------------------------------
            if USE_EVIDENCE or USE_GRAPH:
                preds = get_preds_triplet(
                    logits1=exemplar_logits,
                    logits2=timeline_logits,
                    logits3=extra_logits,
                    combination_type=LOGITS_COMBINATION_TYPE
                )
                
            else:
                preds = get_preds(
                    logits1=exemplar_logits,
                    logits2=timeline_logits,
                    combination_type=LOGITS_COMBINATION_TYPE
                )
            gold = labels.cpu().tolist()

            pred_list.extend(preds)
            gold_list.extend(gold)
  
    return pred_list, gold_list




def train_dml(
    model_info,
    models,
    train_data_loader,
    val_data_loader,
    test_data_loader,
    base_learning_rate,
    classifier_learning_rate,
    weight_decay,
    num_labels,
):
    
    optimizers = prepare_for_training_dml(models=models,
                                          base_learning_rate=base_learning_rate, 
                                          classifier_learning_rate=classifier_learning_rate, 
                                          weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    patience = 0
    
    for epoch in range(MAX_EPOCHS):
        
        # --------------------------------------------------------- Train Set ---------------------------------------------------------
        
        train_loss, train_preds, train_gold = train_epoch_dml(
            models=models,
            optimizers=optimizers,
            data_loader=train_data_loader
        )
            
        train_losses.append(train_loss['loss'])
        train_results = get_prediction_scores(train_preds, train_gold)
        train_cr = classification_report(y_true=train_gold, y_pred=train_preds, output_dict=True)
        train_cm = confusion_matrix(y_true=train_gold, y_pred=train_preds)
        
         # --------------------------------------------------------- Val Set ---------------------------------------------------------
        
        val_loss, val_preds, val_gold = val_epoch_dml(
            models=models,
            data_loader=val_data_loader
        )
        val_losses.append(val_loss['loss'])
        val_results = get_prediction_scores(val_preds, val_gold)
        val_cr = classification_report(y_true=val_gold, y_pred=val_preds, output_dict=True)
        val_cm = confusion_matrix(y_true=val_gold, y_pred=val_preds)
        
        # --------------------------------------------------------- Test Set ---------------------------------------------------------
        
        test_preds, test_gold = test_epoch_dml(models,
                                               test_data_loader)
        test_results = get_prediction_scores(test_preds, test_gold)
        test_cr = classification_report(y_true=test_gold, y_pred=test_preds, output_dict=True)
        test_cm = confusion_matrix(y_true=test_gold, y_pred=test_preds)
        
        print("\nEpoch: ", epoch+1)
        print("train_exemplar_loss: {}\tval_exemplar_loss: {}".format(train_loss['exemplar_loss'], val_loss['exemplar_loss']))
        print("train_timeline_loss: {}\tval_timeline_loss: {}".format(train_loss['timeline_loss'], val_loss['timeline_loss']))
        if USE_EVIDENCE:
            print("train_evidence_loss: {}\tval_evidence_loss: {}".format(train_loss['extra_loss'], val_loss['extra_loss']))
        if USE_GRAPH:
            print("train_graph_loss: {}\tval_graph_loss: {}".format(train_loss['extra_loss'], val_loss['extra_loss']))
        print("train_loss: {}\tval_loss: {}\tmin_val_loss: {}".format(train_loss['loss'], val_loss['loss'], min(val_losses)))
        
        # ----------------------------------------------------------- Train Results --------------------------------------------------------
        
        print("\ntrain_acc: {}\ttrain_precision: {}\ttrain_recall: {}\ttrain_f1: {}".format(train_results['accuracy'], 
                                                                                            train_results['precision'], 
                                                                                            train_results['recall'], 
                                                                                            train_results['f1-score']))
        
        print("\ntrain_hate_precision: {}\ttrain_hate_recall: {}\ttrain_hate_f1: {}".format(train_cr['0']['precision'], 
                                                                                            train_cr['0']['recall'],
                                                                                            train_cr['0']['f1-score']))

        print("\ntrain_offensive_precision: {}\ttrain_offensive_recall: {}\ttrain_offensive_f1: {}".format(train_cr['1']['precision'], 
                                                                                                           train_cr['1']['recall'],
                                                                                                           train_cr['1']['f1-score']))

        print("\ntrain_provocative_precision: {}\ttrain_provocative_recall: {}\ttrain_provocative_f1: {}".format(train_cr['2']['precision'], 
                                                                                                                 train_cr['2']['recall'],
                                                                                                                 train_cr['2']['f1-score']))

        print("\ntrain_control_precision: {}\ttrain_control_recall: {}\ttrain_control_f1: {}".format(train_cr['3']['precision'], 
                                                                                                     train_cr['3']['recall'],
                                                                                                     train_cr['3']['f1-score']))
        print("\nTraining confusion matrix: ", train_cm)

        # ----------------------------------------------------------- Val Results --------------------------------------------------------
        
        print("\n\nval_acc: {}\tval_precision: {}\tval_recall: {}\tval_f1: {}".format(val_results['accuracy'], 
                                                                                      val_results['precision'], 
                                                                                      val_results['recall'], 
                                                                                      val_results['f1-score']))
        
        print("\nval_hate_precision: {}\tval_hate_recall: {}\tval_hate_f1: {}".format(val_cr['0']['precision'], 
                                                                                      val_cr['0']['recall'],
                                                                                      val_cr['0']['f1-score']))

        print("\nval_offensive_precision: {}\tval_offensive_recall: {}\tval_offensive_f1: {}".format(val_cr['1']['precision'], 
                                                                                                     val_cr['1']['recall'],
                                                                                                     val_cr['1']['f1-score']))

        print("\nval_provocative_precision: {}\tval_provocative_recall: {}\tval_provocative_f1: {}".format(val_cr['2']['precision'], 
                                                                                                           val_cr['2']['recall'],
                                                                                                           val_cr['2']['f1-score']))

        print("\nval_control_precision: {}\tval_control_recall: {}\tval_control_f1: {}".format(val_cr['3']['precision'], 
                                                                                               val_cr['3']['recall'],
                                                                                               val_cr['3']['f1-score']))
        print("\nValidation confusion matrix: ", val_cm)
        
        # ----------------------------------------------------------- Test Results --------------------------------------------------------
        
        print("\n\ntest_acc: {}\ttest_precision: {}\ttest_recall: {}\ttest_f1: {}".format(test_results['accuracy'], 
                                                                                          test_results['precision'], 
                                                                                          test_results['recall'], 
                                                                                          test_results['f1-score']))
        
       
        print("\ntest_hate_precision: {}\ttest_hate_recall: {}\ttest_hate_f1: {}".format(test_cr['0']['precision'], 
                                                                                         test_cr['0']['recall'],
                                                                                         test_cr['0']['f1-score']))

        print("\ntest_offensive_precision: {}\ttest_offensive_recall: {}\ttest_offensive_f1: {}".format(test_cr['1']['precision'], 
                                                                                                        test_cr['1']['recall'],
                                                                                                        test_cr['1']['f1-score']))

        print("\ntest_provocative_precision: {}\ttest_provocative_recall: {}\ttest_provocative_f1: {}".format(test_cr['2']['precision'], 
                                                                                                              test_cr['2']['recall'],
                                                                                                              test_cr['2']['f1-score']))

        print("\ntest_control_precision: {}\ttest_control_recall: {}\ttest_control_f1: {}".format(test_cr['3']['precision'], 
                                                                                                  test_cr['3']['recall'],
                                                                                                  test_cr['3']['f1-score']))
        print("\nTest confusion matrix: ", test_cm)
        
        
        # --------------------------------------------- Storing Val Results ---------------------------------------------
        if SAVE_RESULTS:
            val_result_df = pd.DataFrame(list(zip(val_gold, val_preds)), columns=['gold', 'preds'])

            val_folder = '' + MODEL_TYPE + '/' + str(model_info['model_name']) + '/val/' + str(model_info['timestamp']) + '/'

            if not os.path.exists(val_folder):
                print("\n\nCreating folder at: ", val_folder)
                os.makedirs(val_folder)

            val_path = val_folder + str(model_info['model_name']) + '_dml_timestamp_' + str(model_info['timestamp']) +'_val_epoch_' + str(epoch+1) + '.csv'

            val_result_df.to_csv(val_path, index=False)
            print("\nStored val data at: ", val_path)

            # --------------------------------------------- Storing Test Results ---------------------------------------------

            test_result_df = pd.DataFrame(list(zip(test_gold, test_preds)), columns=['gold', 'preds'])

            test_folder = '' + MODEL_TYPE + '/' + str(model_info['model_name']) + '/test/' + str(model_info['timestamp']) + '/'

            if not os.path.exists(test_folder):
                print("\n\nCreating folder at: ", test_folder)
                os.makedirs(test_folder)

            test_path = val_folder + str(model_info['model_name']) + '_dml_timestamp_' + str(model_info['timestamp']) +'_test_epoch_' + str(epoch+1) + '.csv'

            test_result_df.to_csv(test_path, index=False)
            print("\nStored test data at: ", test_path)
        
        del train_loss
        del val_loss
        del train_preds
        del train_gold
        del train_results
        del train_cr
        del val_preds
        del val_gold
        del val_results
        del val_cr
        del test_preds
        del test_gold
        del test_results
        del test_cr
        gc.collect()
        torch.cuda.empty_cache()