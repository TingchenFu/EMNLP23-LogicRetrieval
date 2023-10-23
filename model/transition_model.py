import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from model.uni_transformer import Block
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# transition model
# inference GRU bi-directional transformer 
# generation uni-directional transformer
# prior uni-directional transformer

# scheduling network

class Transition(nn.Module):
    def __init__(self,config,hidden_dim,z_dim,n_facet1=256,n_facet2=64,n_facet3=16,max_turn=15,layer=3,) -> None:
        super(Transition,self).__init__()
        self.hidden_dim=hidden_dim
        self.config=config
        self.gru=nn.GRU(self.hidden_dim,self.hidden_dim)

        self.z_dim=z_dim
        self.n_facet1=n_facet1
        self.n_facet2=n_facet2
        self.n_facet3=n_facet3
        
        self.cluster_embedder_facet1=nn.Embedding(self.n_facet1,self.hidden_dim)
        self.cluster_embedder_facet2=nn.Embedding(self.n_facet2,self.hidden_dim)
        self.cluster_embedder_facet3=nn.Embedding(self.n_facet3,self.hidden_dim)
        
        self.inference_transformer=BertModel(config)
        self.prior_uni_transformer = nn.ModuleList([Block(max_turn+1, hidden_dim ,scale=True) for _ in range(layer)])

        self.gaussian_mu1=nn.Parameter(torch.FloatTensor(self.n_facet1,z_dim))
        self.gaussian_mu2=nn.Parameter(torch.FloatTensor(self.n_facet2,z_dim))
        self.gaussian_mu3=nn.Parameter(torch.FloatTensor(self.n_facet3,z_dim))
        self.gaussian_sigma1=nn.Parameter(torch.FloatTensor(self.n_facet1,z_dim))
        self.gaussian_sigma2=nn.Parameter(torch.FloatTensor(self.n_facet2,z_dim))
        self.gaussian_sigma3=nn.Parameter(torch.FloatTensor(self.n_facet3,z_dim))

        self.classifier_facet1=nn.Linear(self.hidden_dim,self.n_facet1)
        self.classifier_facet2=nn.Linear(self.hidden_dim,self.n_facet2)
        self.classifier_facet3=nn.Linear(self.hidden_dim,self.n_facet3)

        self.ladder_mlp=nn.Linear(2*self.hidden_dim,self.hidden_dim)
        self.gaussian_mlp=nn.Linear(2*self.hidden_dim,2*self.z_dim)

        self.word_embedder=self.inference_transformer.get_input_embeddings()
        self.lm_head=self.inference_transformer.get_input_embeddings()

        self.generative_transformer=BertModel(config)
        # model parameter init

    def encode(self,utterance_id,utterance_mask):
        '''
        utternace_id:
        utterance_mask:
        '''
        #(bs,seq_len,hidden_dim)
        bs=utterance_id.shape[0]
        #(bs)
        real_length=utterance_mask.sum(1)
        utterance_emb=self.word_embedder(utterance_id)
        packed=pack_padded_sequence(utterance_emb.view(bs,-1,self.word_embedder.embedding_dim),lengths=real_length.masked_fill(real_length==0,1)  .cpu(),batch_first=True,enforce_sorted=False)
        gru_out,gru_last=self.gru(packed)
        # print(type(gru_last))
        #(hidden)
        gru_last=gru_last.squeeze(0)
        
        #(bs,n_cluster)
        pi1=self.classifier_facet1(gru_last)
        pi2=self.classifier_facet2(gru_last)
        pi3=self.classifier_facet3(gru_last)
        
        return pi1,pi2,pi3
    
    def inference(self,utterance_id,utterance_mask,):
        '''
        utternace_id:
        utterance_mask:
        '''
        #(bs,seq_len,hidden_dim)
        bs=utterance_id.shape[0]
        #(bs)
        real_length=utterance_mask.sum(1)
        utterance_emb=self.word_embedder(utterance_id)
        packed=pack_padded_sequence(utterance_emb.view(bs,-1,self.word_embedder.embedding_dim),lengths=real_length.masked_fill(real_length==0,1).cpu(),batch_first=True,enforce_sorted=False)
        gru_out,gru_last=self.gru(packed)
        # print(type(gru_last))
        #(hidden)
        gru_last=gru_last.squeeze(0)
        
        #(bs,n_cluster)
        pi1=self.classifier_facet1(gru_last)
        pi2=self.classifier_facet2(gru_last)
        pi3=self.classifier_facet3(gru_last)

        #(bs,n_cluster)
        y1=F.gumbel_softmax(pi1,tau=1.0,hard=True)
        y2=F.gumbel_softmax(pi2,tau=1.0,hard=True)
        y3=F.gumbel_softmax(pi3,tau=1.0,hard=True)

        #(bs,hidden)
        y1_emb=torch.matmul(y1,self.cluster_embedder_facet1.weight)
        y2_emb=torch.matmul(y2,self.cluster_embedder_facet2.weight)
        y3_emb=torch.matmul(y3,self.cluster_embedder_facet3.weight)

        hidden=self.inference_transformer(utterance_id,utterance_mask,return_dict=True,output_hidden_states=True)['hidden_states'][-3:]
        #(bs,hidden)
        h_cls_facet1=hidden[0][:,0,:]
        h_cls_facet2=hidden[1][:,0,:]
        h_cls_facet3=hidden[2][:,0,:]

        v1= h_cls_facet1
        v2= self.ladder_mlp(torch.cat([h_cls_facet2,y1_emb],dim=1))
        v3= self.ladder_mlp(torch.cat([h_cls_facet3,y2_emb],dim=1))

        mu_facet1,sigma_facet1=self.gaussian_mlp(torch.cat([v1,y1_emb],dim=1)).chunk(2,dim=1)
        mu_facet2,sigma_facet2=self.gaussian_mlp(torch.cat([v2,y2_emb],dim=1)).chunk(2,dim=1)
        mu_facet3,sigma_facet3=self.gaussian_mlp(torch.cat([v3,y3_emb],dim=1)).chunk(2,dim=1)

        # print(pi1.shape)
        # print(mu_facet1.shape)
        # print(sigma_facet1.shape)
        # print("here is ok")
        #(bs,n_cluster), (bs,zdim), (bs,zdim)
        return [pi1, pi2, pi3], [mu_facet1, mu_facet2, mu_facet3], [sigma_facet1,sigma_facet2,sigma_facet3]

    def prior(self,context_piy1,context_piy2,context_piy3,turn_mask=None):
        '''
        context_id (bs,n_turn,seq_len)
        context_mask (bs,n_turn,seq_len)
        context_piy (bs,n_turn,n_cluster)  or (bs,n_turn)the posterior inference of each utterance
        '''
        bs,n_turn=context_piy1.shape[0],context_piy1.shape[1]
        cluster_emb_facet1=[]
        cluster_emb_facet2=[]
        cluster_emb_facet3=[]
        
        if context_piy1.dim() == 3:
            assert context_piy2.dim() == 3 and context_piy3.dim() == 3
            for turn in n_turn:
                utterance_pi1 = context_piy1[:,turn,:]
                utterance_pi2 = context_piy2[:,turn,:]
                utterance_pi3 = context_piy3[:,turn,:]
                # utterance_id = context_id[:,turn,:]
                # utterance_mask = context_mask[:,turn,:]
                # post_pis,post_mus,post_sigmas =  self.inference(utterance_id,utterance_mask)
                # #(bs,n_cluster)
                # post_pi1,post_pi2,post_pi3=post_pis[0],post_pis[1],post_pis[2]
                # post_mu1,post_mu2,post_mu3=post_mus[0],post_mus[1],post_mus[2]
                # post_sigma1,post_sigma2,post_sigma3=post_sigmas[0],post_sigmas[1],post_sigmas[2]

                #(bs,hidden)
                cluster_emb_facet1.append(torch.matmul(F.gumbel_softmax(utterance_pi1,tau=1.0,hard=True),self.cluster_embedder_facet1.weight))
                cluster_emb_facet2.append(torch.matmul(F.gumbel_softmax(utterance_pi2,tau=1.0,hard=True),self.cluster_embedder_facet2.weight))
                cluster_emb_facet3.append(torch.matmul(F.gumbel_softmax(utterance_pi3,tau=1.0,hard=True),self.cluster_embedder_facet3.weight))
            
            #(bs,n_turn,hidden)
            cluster_hidden_facet1=torch.stack(cluster_emb_facet1,dim=1)
            cluster_hidden_facet2=torch.stack(cluster_emb_facet2,dim=1)
            cluster_hidden_facet3=torch.stack(cluster_emb_facet3,dim=1)

        elif context_piy1.dim() == 2:
            assert context_piy2.dim() == 2 and context_piy3.dim() == 2
            cluster_hidden_facet1 = self.cluster_embedder_facet1(context_piy1.masked_fill(context_piy1==-100,0))
            cluster_hidden_facet2 = self.cluster_embedder_facet2(context_piy2.masked_fill(context_piy2==-100,0))
            cluster_hidden_facet3 = self.cluster_embedder_facet3(context_piy3.masked_fill(context_piy3==-100,0))

        else:
            raise NotImplementedError

        # print(cluster_hidden_facet1.shape)
        # print(cluster_hidden_facet2.shape)
        # print(cluster_hidden_facet3.shape)

        for i, block in enumerate(self.prior_uni_transformer):
            outputs = block(cluster_hidden_facet1)
            cluster_hidden_facet1 = outputs[0]        
        for i, block in enumerate(self.prior_uni_transformer):    
            outputs = block(cluster_hidden_facet2,)
            cluster_hidden_facet2 = outputs[0]
        for i, block in enumerate(self.prior_uni_transformer):
            outputs = block(cluster_hidden_facet3,)
            cluster_hidden_facet3 = outputs[0]

        #(bs,n_turn,n_cluster)
        prior_pi1=self.classifier_facet1(cluster_hidden_facet1)
        prior_pi2=self.classifier_facet2(cluster_hidden_facet2)
        prior_pi3=self.classifier_facet3(cluster_hidden_facet3)

        #(bs,n_turn,n_cluster)
        prior_y1 = F.gumbel_softmax(prior_pi1,tau=1.0,hard=True,)
        prior_y2 = F.gumbel_softmax(prior_pi2,tau=1.0,hard=True,)
        prior_y3 = F.gumbel_softmax(prior_pi3,tau=1.0,hard=True,)

        #(bs, n_turn, zdim)
        prior_mu1 = torch.matmul(prior_y1,self.gaussian_mu1)
        prior_sigma1=torch.matmul(prior_y1,self.gaussian_sigma1)
        prior_mu2 = torch.matmul(prior_y2,self.gaussian_mu2)
        prior_sigma2 = torch.matmul(prior_y2,self.gaussian_sigma2)
        prior_mu3 = torch.matmul(prior_y3,self.gaussian_mu3)
        prior_sigma3 = torch.matmul(prior_y3,self.gaussian_sigma3)
        
        #(bs,n_turn,n_cluster), (bs,n_turn,zdim), (bs,n_turn,zdim)
        return [prior_pi1,prior_pi2,prior_pi3],[prior_mu1,prior_mu2,prior_mu3],[prior_sigma1,prior_sigma2,prior_sigma3]

    def generate(self,utterance_id,utterance_mask,z1,z2,z3):
        '''
        utterance_id: (bs,seq_len)
        utterance_mask: (bs,seq_len)
        z1,z2,z3(bs)
        '''
        #(bs,seq_len,hidden)
        word_emb=self.word_embedder(utterance_id)
        #(bs,hidden)
        cluster_emb_facet1=self.cluster_embedder_facet1(z1)
        cluster_emb_facet2=self.cluster_embedder_facet2(z2)
        cluster_emb_facet3=self.cluster_embedder_facet3(z3)
        input_emb=word_emb+cluster_emb_facet1[:,None,:]+cluster_emb_facet2[:,None,:]+cluster_emb_facet3[:,None,:]
        #(bs,seq_len,hidden)
        encoded=self.generative_transformer(attention_mask=utterance_mask,input_embs=input_emb,return_dict=True)['last_hidden_state']
        #(bs,seq_len,vocab)
        lm_logit=self.lm_head(encoded)

        shifted_lm_logits = lm_logit[:, :-1, :].contiguous()
        label=torch.where(utterance_mask,utterance_id,-100)
        label = label[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction='sum')
        lm_loss = loss_fct(shifted_lm_logits.view(-1, self.config.vocab_size), label.view(-1))
        return lm_loss