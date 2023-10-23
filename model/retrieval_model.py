import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertPreTrainedModel, BertModel

#the same version as in the polyarr_model.py
class CrossModel(BertPreTrainedModel):
    def __init__(self, config,args):
        super().__init__(config)
        self.ac2fn={
            'gelu':nn.GELU(),
            'tanh':nn.Tanh(),
            'sigmoid':nn.Sigmoid()
        }
        self.n_cluster=args.n_cluster
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(args.dropout)
        # self.dense=nn.Linear(config.hidden_size,config.hidden_size)
        # self.lin=nn.Linear(config.hidden_size,args.n_cluster)
        # self.activation=nn.Tanh()
        self.classifier=nn.Linear(config.hidden_size,1)
        self.gate=nn.Linear(config.hidden_size+args.n_cluster+args.n_cluster,1)
        self.init_weights()
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.xavier_normal_(self.gate.weight)

    def forward(
        self,
        input_id,
        attention_mask,
        segment_id,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        bs,n_cand=input_id.shape[0],input_id.shape[1]
        
        # for SABERT
        
        hidden = self.bert(
            input_id.contiguous().view(bs*n_cand,-1),
            attention_mask.contiguous().view(bs*n_cand,-1),
            token_type_ids=segment_id.contiguous().view(bs*n_cand,-1),
            position_ids=position_ids,
            #switch_ids=switch_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[1]
        # for ums only
        #hidden=hidden[:,0,:]
        hidden = self.dropout(hidden)


        #(bs,n_cand)
        logit=self.classifier(hidden).squeeze(1).contiguous().view(bs,n_cand)
        #logit=F.sigmoid(logit)
        return logit

    def interact(self,input_id,attention_mask,segment_id,prior_logit,post_logit,):
        bs,n_cand=input_id.shape[0],input_id.shape[1]
        n_cluster=prior_logit.shape[1]
        
        # if switch_ids is not None:
        #     switch_ids=switch_ids.contiguous().view(bs*n_cand,-1)
        
        hidden = self.bert(
            input_id.contiguous().view(bs*n_cand,-1),
            attention_mask.contiguous().view(bs*n_cand,-1),
            token_type_ids=segment_id.contiguous().view(bs*n_cand,-1),
            #switch_ids=switch_ids,
            return_dict=False,
        )[1]
        hidden=self.dropout(hidden).view(bs,n_cand,-1)
        gated=self.gate(torch.cat([hidden,prior_logit.unsqueeze(1).expand_as(post_logit),post_logit],dim=2)).squeeze(-1)
        gated=F.sigmoid(gated)
        KL_distance=F.kl_div(F.log_softmax(prior_logit,dim=1).unsqueeze(1).expand_as(post_logit),F.softmax(post_logit,dim=2),reduction='none').sum(dim=2)
        cross_logit=self.classifier(hidden).squeeze(-1).contiguous().view(bs,n_cand)
        score=gated*cross_logit+(1-gated)*KL_distance
        return score

    def solo(self,input_id,attention_mask,segment_id,prior_logit,post_logit,):
        KL_distance=F.kl_div(F.log_softmax(prior_logit,dim=1).unsqueeze(1).expand_as(post_logit),F.softmax(post_logit,dim=2),reduction='none').sum(dim=2)
        return KL_distance
    
    def visualize(
        self,
        input_id,
        attention_mask,
        segment_id,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        bs,n_cand=input_id.shape[0],input_id.shape[1]
        hidden = self.bert(
            input_id.contiguous().view(bs*n_cand,-1),
            attention_mask.contiguous().view(bs*n_cand,-1),
            token_type_ids=segment_id.contiguous().view(bs*n_cand,-1),
            position_ids=position_ids,
            #switch_ids=switch_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[1]
        
        return hidden



class BiModel(BertPreTrainedModel):
    def __init__(self, config,args):
        super().__init__(config)
        self.ac2fn={
            'gelu':nn.GELU(),
            'tanh':nn.Tanh(),
            'sigmoid':nn.Sigmoid()
        }
        self.n_cluster=args.n_cluster
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(args.dropout)
        # self.dense=nn.Linear(config.hidden_size,config.hidden_size)
        # self.lin=nn.Linear(config.hidden_size,args.n_cluster)
        # self.activation=nn.Tanh()
        # self.classifier=nn.Linear(config.hidden_size,1)
        self.init_weights()

    def forward(
        self,
        context_id,
        candidate_id,
        context_mask,
        candidate_mask,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        bs,n_cand=candidate_id.shape[0],candidate_id.shape[1]
        #(bs,hidden_dim)
        context_hidden=self.bert(context_id,context_mask)[1]
        candidate_hidden=self.bert(candidate_id.view(bs*n_cand,-1),candidate_mask.view(bs*n_cand,-1))[1].view(bs,n_cand,-1)
        context_hidden=self.dropout(context_hidden)
        candidate_hidden=self.dropout(candidate_hidden)
        #(bs,n_cand)
        context_hidden=F.normalize(context_hidden,dim=1,p=2.0)
        candidate_hidden=F.normalize(candidate_hidden,dim=2,p=2.0)
        context_hidden=F.tanh(context_hidden)
        candidate_hidden=F.tanh(candidate_hidden)
        logit=(context_hidden.unsqueeze(1)*candidate_hidden).sum(2)
        return logit



class PolyModel(BertPreTrainedModel):
    def __init__(self, config,args):
        super().__init__(config)
        self.ac2fn={
            'gelu':nn.GELU(),
            'tanh':nn.Tanh(),
            'sigmoid':nn.Sigmoid()
        }
        self.n_cluster=args.n_cluster
        self.config = config
        self.bert = BertModel(config)
        self.lin1=nn.Linear(config.hidden_size*4,config.hidden_size)
        self.lin2=nn.Linear(config.hidden_size*4,config.hidden_size)
        self.affline1=nn.Linear(config.hidden_size,self.n_cluster)
        self.affline2=nn.Linear(config.hidden_size,self.n_cluster)
        self.out=nn.Linear(self.n_cluster*2,1)

        self.dropout = nn.Dropout(args.dropout)
        #self.dense=nn.Linear(config.hidden_size,config.hidden_size)
        self.in_activation=self.ac2fn[args.in_activation]
        self.out_activation=self.ac2fn[args.out_activation]
        self.tau=args.tau
        

        self.init_weights()
        nn.init.xavier_normal_(self.lin1.weight)
        nn.init.xavier_normal_(self.lin2.weight)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.xavier_normal_(self.affline1.weight)
        nn.init.xavier_normal_(self.affline2.weight)
        #nn.init.xavier_normal_(self.dense.weight)

    def forward(
        self,
        context_id,
        candidate_id,
        index_id,
        context_mask,
        candidate_mask,
        index_mask,
        context_segment,
        candidate_segment,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        is_cluster=False
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        bs,context_len=context_id.shape
        n_cand=candidate_id.shape[1]
        n_turn=index_id.shape[1]

        #(bs,seq_len,hidden_dim)
        context_hidden = self.bert(
            context_id,
            attention_mask=context_mask,
            token_type_ids=context_segment,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]
        #(bs,n_turn,hidden_dim)
        turn_hidden=torch.gather(context_hidden,dim=1,index=index_id.unsqueeze(-1).expand(bs,n_turn,self.config.hidden_size))
        #(bs,hidden_dim)
        cls_vector=context_hidden[:,0,:]
        #(bs,n_turn,hidden)
        cls_vector=cls_vector.unsqueeze(1).expand_as(turn_hidden)
        #(bs,n_turn,hidden_dim*4)
        turn_hidden=torch.cat([turn_hidden,cls_vector,cls_vector-turn_hidden,cls_vector*turn_hidden],dim=2)

        turn_hidden=self.lin1(turn_hidden)
        turn_hidden=self.in_activation(turn_hidden)
        turn_hidden=self.affline1(turn_hidden)
        # normalize
        #turn_hidden=turn_hidden.contiguous().view(bs*n_turn,self.n_cluster)
        #turn_hidden=(turn_hidden/abs(turn_hidden).max(dim=0)[0].unsqueeze(0)).contiguous().view(bs,n_turn,self.n_cluster)
        #(bs,n_turn,n_cluster)
        turn_hidden=self.out_activation(turn_hidden)

        #(bs*n_cand,seq_len,hidden_dim)
        response_hidden = self.bert(
            candidate_id.view(bs*n_cand,-1),
            attention_mask=candidate_mask.view(bs*n_cand,-1),
            token_type_ids=candidate_segment.unsqueeze(1).unsqueeze(1).repeat(1,n_cand,candidate_id.shape[-1]).contiguous().view(bs*n_cand,-1),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )[0]

        # (bs,n_cand,seq_len,hidden_dim)
        response_hidden=response_hidden.unsqueeze(1).contiguous().view(bs,n_cand,-1,self.config.hidden_size)
        #(bs,n_cand,hidden)
        cls_vector=response_hidden[:,:,0,:]
        #(bs,n_cand,seq_len,hidden)
        mean_hidden=response_hidden*(candidate_mask.unsqueeze(-1).expand_as(response_hidden))
        # (bs,n_cand,hidden_dim)
        mean_hidden=torch.sum(mean_hidden,dim=2)/torch.sum(candidate_mask,dim=2).unsqueeze(-1).float()
        
        candidate_hidden=torch.cat([mean_hidden,cls_vector,cls_vector-mean_hidden,cls_vector*mean_hidden],dim=2)
        candidate_hidden=self.lin2(candidate_hidden)
        candidate_hidden=self.in_activation(candidate_hidden)
        candidate_hidden=self.affline2(candidate_hidden)
        # normalize
        #candidate_hidden=candidate_hidden.contiguous().view(bs*n_cand,self.n_cluster)
        #candidate_hidden=(candidate_hidden/torch.max(abs(candidate_hidden),dim=0)[0].unsqueeze(0)).contiguous().view(bs,n_cand,self.n_cluster)
        #(bs,n_cand,n_cluster)
        candidate_hidden=self.out_activation(candidate_hidden)
        
        # #(n_cluster,bs*n_cand)
        # feature_vector=candidate_hidden.contiguous().view(bs*n_cand,self.n_cluster).transpose(0,1)
        # scaled=F.normalize(feature_vector,dim=1,p=2.0)
        # #(n_cluster,n_cluster)
        # cluster_attn=torch.matmul(scaled,scaled.transpose(0,1))/self.tau
        # #(n_cluster)
        # feature_loss=-torch.log_softmax(cluster_attn,dim=1).diag().mean()
        
        #(bs,n_cand,n_turn)
        attn=torch.matmul(candidate_hidden,turn_hidden.transpose(1,2))/math.sqrt(self.config.hidden_size)
        attn=attn+(~index_mask).unsqueeze(1).expand_as(attn)*-10000.0
        attn=F.softmax(attn,dim=2)

        #(bs,n_cand,hidden_dim)
        context_tensor=torch.matmul(attn,turn_hidden)
        #(bs,n_cand)
        #logit=self.out(torch.cat([candidate_hidden,context_tensor],dim=2)).squeeze(2)
        logit=torch.sum(candidate_hidden*context_tensor,dim=2)
        #normalize
        #logit=logit/(torch.max(logit,dim=1)[0].unsqueeze(-1))
        #logit=F.sigmoid(logit)
        #(bs,n_cand)
        return logit

    def encode(self,input_id,attention_mask):
        bs=input_id.shape[0]
        hidden=self.bert(
            input_id,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
        )[0]
        cls_vector=hidden[:,0,:]
        #(bs,seq_len,hidden)
        mean_hidden=hidden*(attention_mask.unsqueeze(-1).expand_as(hidden))
        # (bs,n_turn,hidden_dim)
        mean_hidden=torch.sum(mean_hidden,dim=2)/torch.sum(attention_mask,dim=2).unsqueeze(-1).float()
        
        rep=torch.cat([mean_hidden,cls_vector,cls_vector-mean_hidden,cls_vector*mean_hidden],dim=2)
        rep=self.lin2(rep)
        rep=self.in_activation(rep)
        rep=self.affline2(rep)
        return rep


class CrossModelv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        # self.dense=nn.Linear(config.hidden_size,config.hidden_size)
        # self.lin=nn.Linear(config.hidden_size,args.n_cluster)
        # self.activation=nn.Tanh()
        self.dropout=nn.Dropout(0.1)
        self.classifier=nn.Linear(config.hidden_size,1)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(
        self,
        input_id, #(bs,n_cand,seq_len)
        attention_mask,
        segment_id,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        bs,n_cand=input_id.shape[0],input_id.shape[1]
        
        # for SABERT
        logit=[]

        hidden = self.bert(
            input_ids=input_id.contiguous().view(bs*n_cand,-1),
            attention_mask=attention_mask.contiguous().view(bs*n_cand,-1),
            token_type_ids=segment_id.contiguous().view(bs*n_cand,-1),
        )[1]
        # for ums only
        #hidden=hidden[:,0,:]
        hidden = self.dropout(hidden)
        #(bs,n_cand)
        logit=self.classifier(hidden).squeeze(1).contiguous().view(bs,n_cand)
        #logit=F.sigmoid(logit)
        return logit