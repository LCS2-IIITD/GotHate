# Code for data utils for HOPN

import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import dgl
from config.config import *

# -------------------------------------------------------------- DATA UTILS --------------------------------------------------------------

def get_graph_data(
    graph_path,
    graph_embedding_path,    
):
    # ------------------------------------- GRAPH EMBEDDINGS -------------------------------------
    
    graph_embeddings = torch.load(graph_embedding_path)
    print("node_embeddings size: ", graph_embeddings.shape)

    start = datetime.now()
    with open(graph_path, 'rb') as file:
        graph_data_dict = pickle.load(file)
    file.close()
    print("graph_data_dict size", len(graph_data_dict))

    dgl_graph_data_dict = dict()
    for uid in graph_data_dict.keys():

        temp = graph_data_dict[uid]

        for node in temp.nodes():
            temp.nodes[node]['node_feat'] = graph_embeddings[node].float().detach().numpy()

        if len(temp.edges()) == 0:
            node = list(temp.nodes())[0]
            temp.add_edge(node, node, edge_weight=1.0)

        temp_dgl = dgl.from_networkx(temp, node_attrs=['node_feat'])
        temp_dgl = dgl.from_networkx(temp, node_attrs=['node_feat'], edge_attrs=['edge_weight'])
        temp_dgl.edata['edge_weight'] = temp_dgl.edata['edge_weight'].to(dtype=torch.float32)
        
    #         if (temp_dgl.in_degrees() == 0).any():
        temp_dgl = dgl.add_self_loop(temp_dgl)
        dgl_graph_data_dict[uid] = temp_dgl
   
    return dgl_graph_data_dict, graph_embeddings



# def get_graph_data(
#     graph_path,
#     graph_embedding_path,    
# ):
#     # ------------------------------------- GRAPH EMBEDDINGS -------------------------------------
    
#     graph_embeddings = torch.load(graph_embedding_path)
#     print("node_embeddings size: ", graph_embeddings.shape)

#     start = datetime.now()
#     with open(graph_path, 'rb') as file:
#         graph_data_dict = pickle.load(file)
#     file.close()
#     print("graph_data_dict size", len(graph_data_dict))

#     dgl_graph_data_dict = dict()
#     for uid in graph_data_dict.keys():
#         temp = graph_data_dict[uid]
#         for node in temp.nodes():
#             temp.nodes[node]['node_feat'] = graph_embeddings[node].detach().numpy()
#         temp_dgl = dgl.from_networkx(temp, node_attrs=['node_feat'])
# #         temp_dgl = dgl.from_networkx(temp, node_attrs=['node_feat'], edge_attrs=['weight'])
# #         if (temp_dgl.in_degrees() == 0).any():
#         temp_dgl = dgl.add_self_loop(temp_dgl)
#         dgl_graph_data_dict[uid] = temp_dgl
   
#     return dgl_graph_data_dict


