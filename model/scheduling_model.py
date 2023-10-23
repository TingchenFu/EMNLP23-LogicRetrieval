import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel

class FFN(nn.Module):
    def __init__(self,input_dim):
        super(FFN,self).__init__()
        self.up=nn.Linear(input_dim, input_dim*4)
        self.down=nn.Linear(input_dim*4, input_dim)
        self.drop=nn.Dropout(p=0.1)
        self.layer_norm=nn.LayerNorm(input_dim)
        self.activation=F.gelu

    def forward(self,input):
        tensor=self.up(input)
        tensor=self.activation(tensor)
        tensor=self.down(tensor)
        tensor=self.drop(tensor)
        tensor=self.layer_norm(tensor+input)

        return tensor


class Scheduler(nn.Module):
    def __init__(self,n_facet1,n_facet2,n_facet3):
        super(Scheduler,self).__init__()
        self.mlp_facet1=FFN(n_facet1)
        self.mlp_facet2=FFN(n_facet2)
        self.mlp_facet3=FFN(n_facet3)

    def forward(self, prior_pi1, prior_pi2, prior_pi3):
        '''
        prior_pi(bs,n_cluster)
        '''
        prior_pi1=self.mlp_facet1(prior_pi1)
        prior_pi2=self.mlp_facet2(prior_pi2)
        prior_pi3=self.mlp_facet3(prior_pi3)
        
        return prior_pi1, prior_pi2, prior_pi3

    def naive(self, prior_pi1, prior_pi2, prior_pi3):
        return prior_pi1, prior_pi2, prior_pi3