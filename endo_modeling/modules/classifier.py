# Code for the classifiers of HOPN

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import *

# ------------------------------------------ Classifier Module ------------------------------------------ #

class ClassifierHead(nn.Module):
    
    def __init__(
        self,
        model_dim: int,
        num_labels: int
    ):
        super(ClassifierHead, self).__init__()
        
        self.transform1 = nn.Linear(model_dim, GRAPH_DIM*3)
        self.dropout1 = nn.Dropout(DROPOUT_RATE)
        self.layer_norm1 = nn.LayerNorm(GRAPH_DIM*3)
        
        self.transform2 = nn.Linear(GRAPH_DIM*3, GRAPH_DIM)
        self.dropout2 = nn.Dropout(DROPOUT_RATE)
        self.layer_norm2 = nn.LayerNorm(GRAPH_DIM)
        
        self.classifier = nn.Linear(GRAPH_DIM, num_labels)
        
            
            
    def forward(
        self,
        input_tensor: torch.Tensor
    ):

        output = F.relu(self.transform1(input_tensor))
        output = self.layer_norm1(self.dropout1(output)) 
        
        output = F.relu(self.transform2(output))
        output = self.layer_norm2(self.dropout2(output)) 
        
        output = self.classifier(output)
        return output

