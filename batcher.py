import torch
import random
from util import overlap_mask,subsequence_mask,substring_mask
class MultiFacetBatcher():
    def __init__(self,tokenizer,max_turn,max_utterance_length,device,max_cross_length=256) -> None:
        self.tokenizer=tokenizer
        self.max_turn=max_turn
        self.max_utterance_length=max_utterance_length
        self.device=device
        self.max_cross_length=max_cross_length

    def encode(self,dialog_list):
        '''
        tokenize a dialogue session
        '''
        batch_corpus_id=[]
        bs = len(dialog_list)
        for i in range(bs):
            batch_corpus_id.extend(self.tokenizer.batch_encode_plus(dialog_list[i],max_length=self.max_utterance_length,truncation=True,padding=False)['input_ids'])
        max_length=max([len(x) for x in batch_corpus_id])
        for i in range(len(batch_corpus_id)):
            padding_length=max_length-len(batch_corpus_id[i])
            if padding_length:
                batch_corpus_id[i].extend([self.tokenizer.pad_token_id]*padding_length)
            assert len(batch_corpus_id[i])==self.max_utterance_length
        batch_corpus_id=torch.tensor(batch_corpus_id,dtype=torch.long,device=self.device)

        return {
            'corpus_id':batch_corpus_id
        }

    def preprocess(self,dialog_list,facet1_list,facet2_list,facet3_list):
        '''
        tokenize a dialog session and pair it with facet it.
        '''
        batch_context_id=[]
        batch_candidate_id=[]
        batch_context_facet1_id=[]
        batch_context_facet2_id=[]
        batch_context_facet3_id=[]

        # batch_candidate_facet1_id=[]
        # batch_candidate_facet2_id=[]
        # batch_candidate_facet3_id=[]
        # print(self.max_turn)
        
        bs = len(dialog_list)
        for i in range(bs):
            batch_context_id.append(self.tokenizer.batch_encode_plus(dialog_list[i][:-2][-self.max_turn:],max_length=self.max_utterance_length,truncation=True,padding='max_length')['input_ids'])
            batch_candidate_id.append(self.tokenizer.batch_encode_plus(dialog_list[i][-2:],max_length=self.max_utterance_length,truncation=True,padding='max_length')['input_ids'])

            batch_context_facet1_id.append(facet1_list[i][:-2][-self.max_turn:])
            batch_context_facet2_id.append(facet2_list[i][:-2][-self.max_turn:])
            batch_context_facet3_id.append(facet3_list[i][:-2][-self.max_turn:])

            # batch_candidate_facet1_id.append(facet1_list[i][-2:])
            # batch_candidate_facet2_id.append(facet2_list[i][-2:])
            # batch_candidate_facet3_id.append(facet3_list[i][-2:])

        longest=max([len(cid) for cid in batch_context_id])
        
        for i in range(bs):
            padding_length = longest-len(batch_context_id[i])
            if padding_length:
                batch_context_facet1_id[i].extend([-100]*padding_length)
                batch_context_facet2_id[i].extend([-100]*padding_length)
                batch_context_facet3_id[i].extend([-100]*padding_length)
                while len(batch_context_id[i])<longest:
                    batch_context_id[i].append([self.tokenizer.pad_token_id]*self.max_utterance_length)
            assert len(batch_context_id[i])==longest
            assert len(batch_context_facet1_id[i])==longest, print(len(batch_context_facet1_id[i]),longest)
        
        batch_context_id = torch.tensor(batch_context_id,dtype=torch.long,device=self.device)
        batch_candidate_id = torch.tensor(batch_candidate_id,dtype=torch.long,device=self.device)
        batch_context_facet1_id = torch.tensor(batch_context_facet1_id,dtype=torch.long,device=self.device)
        batch_context_facet2_id = torch.tensor(batch_context_facet2_id,dtype=torch.long,device=self.device)
        batch_context_facet3_id = torch.tensor(batch_context_facet3_id,dtype=torch.long,device=self.device)


        return {
            #(bs,n_turn,seq_len)
            'context_id':batch_context_id,
            'candidate_id':batch_candidate_id,
            'context_facet1_id':batch_context_facet1_id,
            'context_facet2_id':batch_context_facet2_id,
            'context_facet3_id':batch_context_facet3_id,
            'candidate_facet1_id': torch.tensor([x[-2:] for x in facet1_list],dtype=torch.long,device=self.device),
            'candidate_facet2_id': torch.tensor([x[-2:] for x in facet2_list],dtype=torch.long,device=self.device),
            'candidate_facet3_id': torch.tensor([x[-2:] for x in facet3_list],dtype=torch.long,device=self.device)
        }


    def retrieve(self,context_list,candidate_list,keep=False,n_rand=0,n_inter=0):
        '''
        tokenize for retrieval
        '''
        # dialog list contains the context, the golden response and the original negative
        batch_input_id=[]
        batch_segment_id=[]
        batch_switch_id=[]
        bs=len(context_list)
        

        for i in range(bs):
            batch_input_id.append([])
            batch_segment_id.append([])
            batch_switch_id.append([])
            # the input  contains the interaction between a context and its many candidates
            context_id=self.tokenizer.encode(' [EOS] '.join(context_list[i])+' [EOS] ',add_special_tokens=False)
            switch_id=[0]*len(context_id)
            user=0
            for j in range(len(context_id)):
                switch_id[j]=user
                if context_id[j]==self.tokenizer.eos_token_id:
                    user=1-user
            for candidate in candidate_list[i]:
                candidate_id=self.tokenizer.encode(candidate,add_special_tokens=False)
                while len(context_id)+len(candidate_id)>self.max_cross_length-3:
                    if len(context_id)> len(candidate_id):
                        context_id.pop(0)
                        switch_id.pop(0)
                    else:
                        candidate_id.pop()
                temp_id=[self.tokenizer.cls_token_id]+context_id+[self.tokenizer.sep_token_id]+candidate_id+[self.tokenizer.sep_token_id]
                segment_id=[0]*(len(context_id)+2)+[1]*(len(candidate_id)+1)
                temp_switch_id=[0]+switch_id+ [user]*(len(temp_id)-len(switch_id)-1)
                padding_length=self.max_cross_length-len(temp_id)
                if padding_length>0:
                    temp_id.extend([self.tokenizer.pad_token_id]*padding_length)
                    segment_id.extend([self.tokenizer.pad_token_id]*padding_length)
                    temp_switch_id.extend([self.tokenizer.pad_token_id]*padding_length)
                
                batch_input_id[i].append(temp_id)
                batch_segment_id[i].append(segment_id)
                batch_switch_id[i].append(temp_switch_id)

        #(bs,n_cand,seq_len)
        batch_input_id=torch.tensor(batch_input_id,dtype=torch.long,device=self.device)
        #(bs,n_cand,seq_len)
        batch_segment_id=torch.tensor(batch_segment_id,dtype=torch.long,device=self.device)
        batch_switch_id=torch.tensor(batch_switch_id,dtype=torch.long,device=self.device)
        # print(batch_input_id.shape)
        # print(batch_segment_id.shape)
        # print(batch_switch_id.shape)
        
        return{
            'input_id':batch_input_id,
            'segment_id':batch_segment_id,
            'switch_id':batch_switch_id
        }