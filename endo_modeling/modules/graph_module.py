# Code for the graph module of HOPN

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dgl_nn

from config.config import *
from .attention_module import *

# ------------------------------------------ Graph Module ------------------------------------------ #
    
class GraphModule(nn.Module):
    def __init__(
        self, 
        model_dim: int
    ):
        super(GraphModule, self).__init__()
        self.conv1 = dgl_nn.GraphConv(GRAPH_DIM, GRAPH_DIM*2)
        self.conv2 = dgl_nn.GraphConv(GRAPH_DIM*2, GRAPH_DIM*3)
        self.conv3 = dgl_nn.GraphConv(GRAPH_DIM*3, GRAPH_DIM*4)

        self.pooling = dgl_nn.MaxPooling()
        self.graph_len_transform = nn.Linear(GRAPH_SEQ_DIM, SEQ_DIM)
        self.graph_dropout = nn.Dropout(DROPOUT_RATE)
#         self.graph_attention_layer = MultiHeadAttention(d_model=GRAPH_DIM*4, num_heads=NUM_HEADS)
        self.graph_attention_layer = ScaledDotProductAttention(dim=GRAPH_DIM*4)
        
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
        lm_tensor,
        graph
    ):
        # 1. Obtain node embeddings 
        node_feat = graph.ndata['node_feat']
        
        if USE_GRAPH_EDGE:
            edge_weight = graph.edata['edge_weight']
            graph_output = F.relu(self.conv1(graph, node_feat, edge_weight=edge_weight))
            graph_output = F.relu(self.conv2(graph, graph_output, edge_weight=edge_weight))
            graph_output = F.relu(self.conv3(graph, graph_output, edge_weight=edge_weight))
        
        else:
            graph_output = F.relu(self.conv1(graph, node_feat))
            graph_output = F.relu(self.conv2(graph, graph_output))
            graph_output = F.relu(self.conv3(graph, graph_output))
        
        # 2. Apply pooling
        graph_output = F.relu(self.pooling(graph, graph_output))
        
        # 3. Convert to 3D tensor
        graph_output = graph_output.unsqueeze(1).repeat(1, GRAPH_SEQ_DIM, 1)
        graph_output = graph_output.permute(0, 2, 1)
        graph_output = self.graph_len_transform(graph_output)
        graph_output = graph_output.permute(0, 2, 1)
        graph_output = self.graph_dropout(graph_output)
        
        graph_output, _ = self.graph_attention_layer(query=graph_output, 
                                                     key=graph_output, 
                                                     value=graph_output)
        
        # 4. Fuse with LM output
        lm_tensor = self.lm_dropout(F.relu(self.lm_transform(lm_tensor)))
        lm_tensor, _ = self.lm_attention_layer(query=lm_tensor, 
                                               key=lm_tensor, 
                                               value=lm_tensor)
        
        output, _ = self.cross_attention_layer(query=graph_output, 
                                               key=lm_tensor, 
                                               value=lm_tensor)
    
        output = F.relu(self.final_transform(output))
        output = self.layer_norm(output)
        
        return output
