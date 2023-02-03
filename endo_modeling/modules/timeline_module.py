# Code for the timeline module of HOPN

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import *
from .attention_module import *

# ------------------------------------------ Timeline Module ------------------------------------------ #

# class TimelineModule(nn.Module):
    
#     def __init__(
#         self,
#         model_dim: int
#     ):
#         super(TimelineModule, self).__init__()
        
#         self.len_transform = nn.Linear(TIMELINE_SEQ_DIM, SEQ_DIM)
#         self.timeline_transform = nn.Linear(model_dim, GRAPH_DIM*4)
#         self.timeline_dropout = nn.Dropout(DROPOUT_RATE)
# #         self.bilstm = nn.LSTM(
# #             input_size=GRAPH_DIM*4,
# #             hidden_size=GRAPH_DIM*4,
# #             num_layers=2,
# #             bias=True,
# #             batch_first=True,
# #             dropout=DROPOUT_RATE,
# #             bidirectional=True
# #         )
        
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
#         timeline_input: torch.Tensor
#     ):
#         timeline_input = timeline_input.permute(0, 2, 1)
#         timeline_input = self.len_transform(timeline_input)
#         timeline_input = timeline_input.permute(0, 2, 1)
#         timeline_input = self.timeline_dropout(self.timeline_transform(timeline_input))
# #         timeline_input, _, _ = self.bilstm(timeline_input)
        
#         lm_tensor = self.lm_dropout(F.relu(self.lm_transform(lm_tensor)))
#         output, _ = self.attention_layer(query=timeline_input.permute(1, 0, 2),
#                                          key=lm_tensor.permute(1, 0, 2), 
#                                          value=lm_tensor.permute(1, 0, 2))
#         output = F.relu(self.final_transform(output.permute(1, 0, 2)))
#         output = self.layer_norm(output)
#         return output
    
    
# ------------------------------------------ Timeline Module ------------------------------------------ #

class TimelineModule(nn.Module):
    
    def __init__(
        self,
        model_dim: int
    ):
        super(TimelineModule, self).__init__()
        
        self.timeline_len_transform = nn.Linear(TIMELINE_SEQ_DIM, SEQ_DIM)
        self.timeline_transform = nn.LSTM(
            input_size=model_dim,
            hidden_size=GRAPH_DIM*2,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=DROPOUT_RATE,
            bidirectional=True
        )
        self.timeline_dropout = nn.Dropout(DROPOUT_RATE)
#         self.timeline_attention_layer = MultiHeadAttention(d_model=GRAPH_DIM*4, num_heads=NUM_HEADS)
        self.timeline_attention_layer = ScaledDotProductAttention(dim=GRAPH_DIM*4)
        
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
        timeline_input: torch.Tensor
    ):
        timeline_input = timeline_input.permute(0, 2, 1)
        timeline_input = self.timeline_len_transform(timeline_input)
        timeline_input = timeline_input.permute(0, 2, 1)
        timeline_input, _ = self.timeline_transform(timeline_input)
        timeline_input = self.timeline_dropout(timeline_input)
        timeline_input, _ = self.timeline_attention_layer(query=timeline_input, 
                                                          key=timeline_input, 
                                                          value=timeline_input)
        
        lm_tensor = self.lm_dropout(F.relu(self.lm_transform(lm_tensor)))
        lm_tensor, _ = self.lm_attention_layer(query=lm_tensor, 
                                               key=lm_tensor, 
                                               value=lm_tensor)
        
        output, _ = self.cross_attention_layer(query=timeline_input, 
                                               key=lm_tensor, 
                                               value=lm_tensor)

        output = F.relu(self.final_transform(output))
        output = self.layer_norm(output)
        return output   




# class TimelineModule(nn.Module):
    
#     def __init__(
#         self,
#         model_dim: int
#     ):
#         super(TimelineModule, self).__init__()
        
#         self.timeline_len_transform = nn.Linear(TIMELINE_SEQ_DIM, SEQ_DIM)
#         self.timeline_transform = nn.LSTM(
#             input_size=model_dim,
#             hidden_size=GRAPH_DIM*2,
#             num_layers=2,
#             bias=True,
#             batch_first=True,
#             dropout=DROPOUT_RATE,
#             bidirectional=True
#         )
#         self.timeline_dropout = nn.Dropout(DROPOUT_RATE)
        
#         self.lm_transform = nn.Linear(model_dim, GRAPH_DIM*4)
#         self.lm_dropout = nn.Dropout(DROPOUT_RATE)
#         self.lm_attention_layer = ScaledDotProductAttention(dim=GRAPH_DIM*4)
        
#         self.cross_attention_layer = ScaledDotProductAttention(dim=GRAPH_DIM*4)
        
#         self.final_transform = nn.Linear(GRAPH_DIM*4, model_dim)
#         self.layer_norm = nn.LayerNorm(model_dim)
        
        
            
#     def forward(
#         self,
#         lm_tensor: torch.Tensor,
#         timeline_input: torch.Tensor
#     ):
#         timeline_input, _ = self.timeline_transform(timeline_input)
#         timeline_input = timeline_input.permute(0, 2, 1)
#         timeline_input = self.timeline_len_transform(timeline_input)
#         timeline_input = timeline_input.permute(0, 2, 1)
#         timeline_input = self.timeline_dropout(timeline_input)
        
#         lm_tensor = self.lm_dropout(F.relu(self.lm_transform(lm_tensor)))
#         lm_tensor, lm_attention = self.lm_attention_layer(query=lm_tensor, 
#                                                           key=lm_tensor, 
#                                                           value=lm_tensor)
        
#         output, timeline_attention = self.cross_attention_layer(query=timeline_input, 
#                                                                 key=lm_tensor, 
#                                                                 value=lm_tensor)
#         output = F.relu(self.final_transform(output))
#         output = self.layer_norm(output)
# #         return output
#         return output, timeline_attention   