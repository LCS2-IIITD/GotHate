# Code for modified bert architecture for HOPN

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertEmbeddings,
    BertPooler,
    BertLayer
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput
)

from modules.graph_module import GraphModule
from modules.exemplar_module import ExemplarModule
from modules.timeline_module import TimelineModule
from modules.evidence_module import EvidenceModule
from modules.fusion_module import *
from modules.dml_mixer import DMLMixer
from config.config import *
from utils.losses import MultiFocalLoss

# ---------------------------------------------- HateBertEncoder ---------------------------------------------- #

class HateBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        
        # ---------------------------------------------- NEW ADDITION ----------------------------------------------
        
        if USE_GRAPH:
            print("\nUSE_GRAPH in BERT\n")
            self.graph_module = GraphModule(model_dim=config.hidden_size)
            
        if USE_EXEMPLAR:
            print("\nUSE_EXEMPLAR in BERT\n")
            self.exemplar_module = ExemplarModule(model_dim=config.hidden_size)
            
        if USE_TIMELINE:
            print("\nUSE_TIMELINE in BERT\n")
            self.timeline_module = TimelineModule(model_dim=config.hidden_size)
            
        if USE_EVIDENCE:
            print("\nUSE_EVIDENCE in BERT\n")
            self.evidence_module = EvidenceModule(model_dim=config.hidden_size)
        
        if MODEL_TYPE == 'fusion':
            if FUSION_TYPE == 'late':
                print("\n\nImplementing Late Fusion in BERT\n\n")
                self.fusion_module = LateFusionModule(model_dim=config.hidden_size)
            elif FUSION_TYPE == 'simple':
                print("\n\nImplementing Simple Fusion in BERT\n\n")
                self.fusion_module = SimpleFusionModule(model_dim=config.hidden_size)
            elif FUSION_TYPE == 'attention':
                print("\n\nImplementing Attention Fusion in BERT\n\n")
                self.fusion_module = AttentionFusionModule(model_dim=config.hidden_size)
            elif FUSION_TYPE == 'stacked-attention':
                print("\n\nImplementing Stacked Attention Fusion in BERT\n\n")
                self.fusion_module = StackedAttentionFusionModule(model_dim=config.hidden_size)
            
            
        # ----------------------------------------------------------------------------------------------------------

        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        graph_input: Optional = None,     # New addition of graph_input
        exemplar_input: Optional[torch.Tensor] = None,    # New Input of exemplar_input
        timeline_input: Optional[torch.Tensor] = None,    # New Input of timeline_input
        evidence_input: Optional[torch.Tensor] = None,    # New Input of evidence_input
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            
            # ---------------------------------------------- NEW ADDITION ----------------------------------------------
            
            if i == FUSION_LAYER:
                feature_set = []
                
                # Exmeplar Module
                if USE_EXEMPLAR:
                    exemplar_output = self.exemplar_module(
                        lm_tensor=hidden_states,
                        exemplar_input=exemplar_input
                    )
                    feature_set.append(exemplar_output)
                
                # Timeline Module
                if USE_TIMELINE:
                    timeline_output = self.timeline_module(
                        lm_tensor=hidden_states,
                        timeline_input=timeline_input
                    )
                    feature_set.append(timeline_output)
                    
                # Graph Module
                if USE_GRAPH:
                    graph_output = self.graph_module(
                        lm_tensor=hidden_states,
                        graph=graph_input
                    )
                    feature_set.append(graph_output)
                    
                # Evidence Module    
                if USE_EVIDENCE:
                    evidence_output = self.evidence_module(
                        lm_tensor=hidden_states,
                        evidence_input=evidence_input
                    ) 
                    feature_set.append(evidence_output)
                
                if MODEL_TYPE == 'fusion':
                    combined_output = self.fusion_module(
                        lm_tensor=hidden_states,
                        feature_inputs=feature_set
                    )
                    hidden_states = combined_output
                                            
                elif MODEL_TYPE == 'dml':
                    output = torch.cat(feature_set, dim=1)
                    hidden_states = torch.cat([output, hidden_states], dim=1)
                   
            # ----------------------------------------------------------------------------------------------------------
        
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    
    
    
# ---------------------------------------------- HateBertModel ---------------------------------------------- #

class HateBertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = HateBertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
        
        
        
        

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        graph_input: Optional = None,     # New addition of graph_input
        exemplar_input: Optional[torch.Tensor] = None,    # New Input of exemplar_input
        timeline_input: Optional[torch.Tensor] = None,    # New Input of timeline_input
        evidence_input: Optional[torch.Tensor] = None,    # New Input of evidence_input
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            graph_input=graph_input,     # New addition of graph_input
            exemplar_input=exemplar_input,    # New Input of exemplar_input
            timeline_input=timeline_input,    # New Input of timeline_input
            evidence_input=evidence_input,    # New Input of evidence_input
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # ---------------------------------------------- NEW ADDITION ----------------------------------------------
        
        if MODEL_TYPE == 'fusion':
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        elif MODEL_TYPE == 'dml':
            exemplar_output = encoder_outputs[0][:, :SEQ_DIM, :]
            timeline_output = encoder_outputs[0][:, SEQ_DIM:SEQ_DIM*2, :]
            sequence_output = encoder_outputs[0][:, SEQ_DIM*2:SEQ_DIM*3, :]

            exemplar_pooled_output = self.pooler(exemplar_output) if self.pooler is not None else None
            timeline_pooled_output = self.pooler(timeline_output) if self.pooler is not None else None
            sequence_pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
            
            pooled_output = torch.cat([exemplar_pooled_output, timeline_pooled_output, sequence_pooled_output], dim=1)
            
            if USE_GRAPH:
                graph_output = encoder_outputs[0][:, SEQ_DIM*3:SEQ_DIM*4, :]
                graph_pooled_output = self.pooler(graph_output) if self.pooler is not None else None
                pooled_output = torch.cat([exemplar_pooled_output, timeline_pooled_output, graph_pooled_output, sequence_pooled_output], dim=1)
                
            elif USE_EVIDENCE:
                evidence_output = encoder_outputs[0][:, SEQ_DIM*3:SEQ_DIM*4, :]
                evidence_pooled_output = self.pooler(evidence_output) if self.pooler is not None else None
                pooled_output = torch.cat([exemplar_pooled_output, timeline_pooled_output, evidence_pooled_output, sequence_pooled_output], dim=1)

        # ----------------------------------------------------------------------------------------------------------

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
#             last_hidden_state=sequence_output,
            last_hidden_state=encoder_outputs[0],
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    
    
# ---------------------------------------------- BertHateClassifier ---------------------------------------------- #

class BertHateClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = HateBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        graph_input: Optional = None,     # New addition of graph_input
        exemplar_input: Optional[torch.Tensor] = None,    # New Input of exemplar_input
        timeline_input: Optional[torch.Tensor] = None,    # New Input of timeline_input
        evidence_input: Optional[torch.Tensor] = None,    # New Input of evidence_input
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            graph_input=graph_input,     # New addition of graph_input
            exemplar_input=exemplar_input,    # New Input of exemplar_input
            timeline_input=timeline_input,    # New Input of timeline_input
            evidence_input=evidence_input,    # New Input of evidence_input
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )