# Code for the fusion modules of HOPN

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import *
from .attention_module import *

# ------------------------------------------ Late Fusion Module ------------------------------------------ #

class LateFusionModule(nn.Module):
    
    def __init__(
        self,
        model_dim: int
    ):
        super(LateFusionModule, self).__init__()
        
        self.len_transform = nn.Linear(TWEET_MAX_LEN, SEQ_DIM)
        
        self.transform1 = nn.Linear(model_dim*3, model_dim)
        self.dropout1 = nn.Dropout(DROPOUT_RATE)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        
            
            
    def forward(
        self,
        exemplar_tensor: torch.Tensor,
        timeline_tensor: torch.Tensor,
        lm_tensor: torch.Tensor
    ):
        lm_tensor = lm_tensor.permute(0, 2, 1)
        lm_tensor = self.len_transform(lm_tensor)
        lm_tensor = lm_tensor.permute(0, 2, 1)
        
        output = F.relu(self.transform1(torch.cat([exemplar_tensor, timeline_tensor, lm_tensor], dim=-1)))
        output = self.layer_norm1(self.dropout1(output)) 

        return output
    
    
    
# ------------------------------------------ Intermediate Fusion Module ------------------------------------------ #

class SimpleFusionModule(nn.Module):
    
    def __init__(
        self,
        model_dim: int
    ):
        super(SimpleFusionModule, self).__init__()
        
        self.len_downsample = nn.Linear(TWEET_MAX_LEN, SEQ_DIM)
        self.len_upsample = nn.Linear(SEQ_DIM, TWEET_MAX_LEN)
        
        if USE_EVIDENCE or USE_GRAPH:
            print("USE_EVIDENCE: {}, USE_GRAPH: {} in SimpleFusionModule\n\n".format(USE_EVIDENCE, USE_GRAPH))
            self.transform1 = nn.Linear(model_dim*4, model_dim)
        else:
            self.transform1 = nn.Linear(model_dim*3, model_dim)
            
        self.dropout1 = nn.Dropout(DROPOUT_RATE)
        self.layer_norm1 = nn.LayerNorm(model_dim)

        
            
            
    def forward(
        self,
        exemplar_tensor: Optional[torch.Tensor] = None,
        timeline_tensor: Optional[torch.Tensor] = None,
        graph_tensor: Optional[torch.Tensor] = None,
        evidence_tensor: Optional[torch.Tensor] = None,
        lm_tensor: Optional[torch.Tensor] = None
    ):
        lm_tensor = lm_tensor.permute(0, 2, 1)
        lm_tensor = self.len_downsample(lm_tensor)
        lm_tensor = lm_tensor.permute(0, 2, 1)
        
        if USE_GRAPH and graph_tensor is not None:
            output = F.relu(self.transform1(torch.cat([exemplar_tensor, timeline_tensor, graph_tensor, lm_tensor], dim=-1)))
            
        elif USE_EVIDENCE and evidence_tensor is not None:
            output = F.relu(self.transform1(torch.cat([exemplar_tensor, timeline_tensor, evidence_tensor, lm_tensor], dim=-1)))
        else:
            output = F.relu(self.transform1(torch.cat([exemplar_tensor, timeline_tensor, lm_tensor], dim=-1)))
        output = self.layer_norm1(self.dropout1(output)) 
                
        output = output.permute(0, 2, 1)
        output = self.len_upsample(output)
        output = output.permute(0, 2, 1)
                        
        return output
    
    
# ------------------------------------------ Attention Fusion Module ------------------------------------------ #
    
# class AttentionFusionModule(nn.Module):
    
#     def __init__(
#         self,
#         model_dim: int
#     ):
#         super(AttentionFusionModule, self).__init__()
        
#         self.len_downsample = nn.Linear(TWEET_MAX_LEN, SEQ_DIM)
#         self.len_upsample = nn.Linear(SEQ_DIM, TWEET_MAX_LEN)
        
#         if USE_EVIDENCE or USE_GRAPH:
#             print("USE_EVIDENCE: {}, USE_GRAPH: {} in AttentionFusionModule\n\n".format(USE_EVIDENCE, USE_GRAPH))
#             self.transform1 = nn.Linear(model_dim*4, model_dim)
#         else:
#             self.transform1 = nn.Linear(model_dim*3, model_dim)
            
#         self.dropout1 = nn.Dropout(DROPOUT_RATE)
#         self.layer_norm1 = nn.LayerNorm(model_dim)
        
#         self.attention_layer = MultiHeadAttention(d_model=model_dim, num_heads=NUM_HEADS)

        
            
            
#     def forward(
#         self,
#         exemplar_tensor: Optional[torch.Tensor] = None,
#         timeline_tensor: Optional[torch.Tensor] = None,
#         graph_tensor: Optional[torch.Tensor] = None,
#         evidence_tensor: Optional[torch.Tensor] = None,
#         lm_tensor: Optional[torch.Tensor] = None
#     ):
#         lm_tensor = lm_tensor.permute(0, 2, 1)
#         lm_tensor = self.len_downsample(lm_tensor)
#         lm_tensor = lm_tensor.permute(0, 2, 1)
        
#         if USE_GRAPH and graph_tensor is not None:
#             output = F.relu(self.transform1(torch.cat([exemplar_tensor, timeline_tensor, graph_tensor, lm_tensor], dim=-1)))
            
#         elif USE_EVIDENCE and evidence_tensor is not None:
#             output = F.relu(self.transform1(torch.cat([exemplar_tensor, timeline_tensor, evidence_tensor, lm_tensor], dim=-1)))
#         else:
#             output = F.relu(self.transform1(torch.cat([exemplar_tensor, timeline_tensor, lm_tensor], dim=-1)))
#         output = self.layer_norm1(self.dropout1(output)) 
        
#         output, _ = self.attention_layer(key=output, query=output, value=output)
                
#         output = output.permute(0, 2, 1)
#         output = self.len_upsample(output)
#         output = output.permute(0, 2, 1)
                        
#         return output
    

    
class AttentionFusionModule(nn.Module):
    
    def __init__(
        self,
        model_dim: int
    ):
        super(AttentionFusionModule, self).__init__()
        
        self.len_downsample = nn.Linear(TWEET_MAX_LEN, SEQ_DIM)
        self.len_upsample = nn.Linear(SEQ_DIM, TWEET_MAX_LEN)
        
        total_dim = model_dim
        if USE_GRAPH:
            print("\nUSE_GRAPH in AttentionFusionModule\n")
            total_dim = total_dim + model_dim
            
        if USE_EXEMPLAR:
            print("\nUSE_EXEMPLAR in AttentionFusionModule\n")
            total_dim = total_dim + model_dim
            
        if USE_TIMELINE:
            print("\nUSE_TIMELINE in AttentionFusionModule\n")
            total_dim = total_dim + model_dim
            
        if USE_EVIDENCE:
            print("\nUSE_EVIDENCE in AttentionFusionModule\n")
            total_dim = total_dim + model_dim
        
        print("\ntotal_dim: ", total_dim, "\n\n")
        self.transform1 = nn.Linear(total_dim, model_dim)
            
        self.dropout1 = nn.Dropout(DROPOUT_RATE)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        
        self.attention_layer = MultiHeadAttention(d_model=model_dim, num_heads=NUM_HEADS)
#         self.attention_layer = ScaledDotProductAttention(dim=GRAPH_DIM*4)

        
            
            
    def forward(
        self,
        lm_tensor: torch.Tensor,
        feature_inputs: list
    ):
        lm_tensor = lm_tensor.permute(0, 2, 1)
        lm_tensor = self.len_downsample(lm_tensor)
        lm_tensor = lm_tensor.permute(0, 2, 1)
        
        output = torch.cat(feature_inputs, dim=-1)
        output = torch.cat([output, lm_tensor], dim=-1)
        
        output = F.relu(self.transform1(output))
        output = self.layer_norm1(self.dropout1(output)) 
        
        output, _ = self.attention_layer(key=output, query=output, value=output)
                
        output = output.permute(0, 2, 1)
        output = self.len_upsample(output)
        output = output.permute(0, 2, 1)
                        
        return output
    
    
# ------------------------------------------ Stacked Attention Fusion Module ------------------------------------------ #
    
class StackedAttentionFusionModule(nn.Module):
    
    def __init__(
        self,
        model_dim: int
    ):
        super(StackedAttentionFusionModule, self).__init__()
        
        self.len_downsample = nn.Linear(TWEET_MAX_LEN, SEQ_DIM)
        
        self.attention_layer = MultiHeadAttention(d_model=model_dim, num_heads=NUM_HEADS)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.layer_norm = nn.LayerNorm(model_dim)

        
            
            
    def forward(
        self,
        exemplar_tensor: Optional[torch.Tensor] = None,
        timeline_tensor: Optional[torch.Tensor] = None,
        graph_tensor: Optional[torch.Tensor] = None,
        evidence_tensor: Optional[torch.Tensor] = None,
        lm_tensor: Optional[torch.Tensor] = None
    ):
        lm_tensor = lm_tensor.permute(0, 2, 1)
        lm_tensor = self.len_downsample(lm_tensor)
        lm_tensor = lm_tensor.permute(0, 2, 1)
        
        if USE_GRAPH and graph_tensor is not None:
            output = torch.cat([exemplar_tensor, timeline_tensor, graph_tensor, lm_tensor], dim=1)
            
        elif USE_EVIDENCE and evidence_tensor is not None:
            output = torch.cat([exemplar_tensor, timeline_tensor, evidence_tensor, lm_tensor], dim=1)
        else:
            output = torch.cat([exemplar_tensor, timeline_tensor, lm_tensor], dim=1)
        
        output, _ = self.attention_layer(key=output, query=output, value=output)
        output = self.layer_norm(self.dropout(output))
                        
        return output
    