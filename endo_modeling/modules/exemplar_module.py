# Code for the exemplar module of HOPN

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import *
from .attention_module import *

# ------------------------------------------ Exemplar Module ------------------------------------------ #

# class ExemplarModule(nn.Module):
    
#     def __init__(
#         self,
#         model_dim: int
#     ):
#         super(ExemplarModule, self).__init__()
        
#         self.len_transform = nn.Linear(EXEMPLAR_SEQ_DIM, SEQ_DIM)
#         self.exemplar_transform = nn.Linear(model_dim, GRAPH_DIM*4)
#         self.exemplar_dropout = nn.Dropout(DROPOUT_RATE)
        
#         self.lm_transform = nn.Linear(model_dim, GRAPH_DIM*4)
#         self.lm_dropout = nn.Dropout(DROPOUT_RATE)
        
#         self.attention_layer = nn.MultiheadAttention(
#             embed_dim=GRAPH_DIM*4, 
#             num_heads=4, 
#             dropout=DROPOUT_RATE, 
#             bias=True,
#             add_zero_attn=False,
# #             batch_first=True
#         )
        
#         self.final_transform = nn.Linear(GRAPH_DIM*4, model_dim)
#         self.layer_norm = nn.LayerNorm(model_dim)
        
        
             
#     def forward(
#         self,
#         lm_tensor: torch.Tensor,
#         exemplar_input: torch.Tensor
#     ):
#         exemplar_input = exemplar_input.permute(0, 2, 1)
#         exemplar_input = self.len_transform(exemplar_input)
#         exemplar_input = exemplar_input.permute(0, 2, 1)
#         exemplar_input = self.exemplar_dropout(self.exemplar_transform(exemplar_input))
        
#         lm_tensor = self.lm_dropout(F.relu(self.lm_transform(lm_tensor)))
#         output, _ = self.attention_layer(query=exemplar_input.permute(1, 0, 2),
#                                          key=lm_tensor.permute(1, 0, 2), 
#                                          value=lm_tensor.permute(1, 0, 2))
#         output = F.relu(self.final_transform(output.permute(1, 0, 2)))
#         output = self.layer_norm(output)
#         return output
    
# ------------------------------------------ Exemplar Module with custom attention ------------------------------------------ #

class ExemplarModule(nn.Module):
    
    def __init__(
        self,
        model_dim: int
    ):
        super(ExemplarModule, self).__init__()
        
        self.exemplar_len_transform = nn.Linear(EXEMPLAR_SEQ_DIM, SEQ_DIM)
        self.exemplar_transform = nn.Linear(model_dim, GRAPH_DIM*4)
        self.exemplar_dropout = nn.Dropout(DROPOUT_RATE)
#         self.exemplar_attention_layer = MultiHeadAttention(d_model=GRAPH_DIM*4, num_heads=NUM_HEADS)
        self.exemplar_attention_layer = ScaledDotProductAttention(dim=GRAPH_DIM*4)
        
        self.lm_transform = nn.Linear(model_dim, GRAPH_DIM*4)
        self.lm_dropout = nn.Dropout(DROPOUT_RATE)
#         self.lm_attention_layer = MultiHeadAttention(d_model=GRAPH_DIM*4, num_heads=NUM_HEADS)
        self.lm_attention_layer = ScaledDotProductAttention(dim=GRAPH_DIM*4)
        
        self.cross_attention_layer = MultiHeadAttention(d_model=GRAPH_DIM*4, num_heads=NUM_HEADS)
#         self.cross_attention_layer = ScaledDotProductAttention(dim=GRAPH_DIM*4)
    
        self.final_transform = nn.Linear(GRAPH_DIM*4, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
        
        
             
    def forward(
        self,
        lm_tensor: torch.Tensor,
        exemplar_input: torch.Tensor
    ):
        exemplar_input = exemplar_input.permute(0, 2, 1)
        exemplar_input = self.exemplar_len_transform(exemplar_input)
        exemplar_input = exemplar_input.permute(0, 2, 1)
        exemplar_input = self.exemplar_dropout(self.exemplar_transform(exemplar_input))
        exemplar_input, _ = self.exemplar_attention_layer(query=exemplar_input, 
                                                          key=exemplar_input, 
                                                          value=exemplar_input)
        
        lm_tensor = self.lm_dropout(F.relu(self.lm_transform(lm_tensor)))
        lm_tensor, _ = self.lm_attention_layer(query=lm_tensor, 
                                               key=lm_tensor, 
                                               value=lm_tensor)
        
        output, _ = self.cross_attention_layer(query=exemplar_input, 
                                               key=lm_tensor, 
                                               value=lm_tensor)

        output = F.relu(self.final_transform(output))
        output = self.layer_norm(output)
        return output   
