import json
from torch.utils.data import Dataset
import random
import pickle
from itertools import product

from torch.utils.data.sampler import BatchSampler

class InferenceDataset(Dataset):
    '''
    only for evaluation, not for training
    '''
    def __init__(self,file,debug=False) -> None:
        super(InferenceDataset,self).__init__()
        self.examples=[]
        f=open(file,mode='r',encoding='utf-8')
        for line in f.readlines():
            data=json.loads(line)
            if 'label' not in data.keys():
                data['label']=[0]*len(data['candidate'])
                data['label'][0]=1
            self.examples.append({
                'context':data['context'],
                'candidate':data['candidate'],
                'label':data['label']
            })
            if debug and  len(self.examples)>=128:
                break
        
    def __len__(self,):
        return len(self.examples)

    def __getitem__(self, index):
        example=self.examples[index]
        return example['context'],example['candidate'],example['label']
    
    @staticmethod
    def collate_fn(batch):
        context_list=[item[0] for item in batch]
        response_list=[item[1] for item in batch]
        label_list=[item[2]for item in batch] 
        return context_list,response_list,label_list

class AdversarialDataset(Dataset):
    def __init__(self,file,debug=False) -> None:
        super(AdversarialDataset,self).__init__()
        self.examples=[]
        f=open(file,mode='r')
        for line in f.readlines():
            data=json.loads(line)
            if 'label' not in data.keys():
                data['label']=[0]*len(data['candidate'])
                data['label'][0]=1
            negative=random.choices(data['context'],k=9)
            self.examples.append({
                'context':data['context'],
                'candidate':data['candidate'][0:1]+negative,
                'label':data['label']
            })
            if debug and len(self.examples)>=1000:
                break
        
    def __len__(self,):
        return len(self.examples)

    def __getitem__(self, index):
        example=self.examples[index]
        return example['context'],example['candidate'],example['label']
    
    @staticmethod
    def collate_fn(batch):
        context_list=[item[0] for item in batch]
        response_list=[item[1] for item in batch]
        label_list=[item[2]for item in batch] 
        return context_list,response_list,label_list



class MultiFacetExample():
    def __init__(self,dialog,cluster_facet1,cluster_facet2,cluster_facet3) -> None:
        self.dialog=dialog
        self.cluster_facet1=cluster_facet1
        self.cluster_facet2=cluster_facet2
        self.cluster_facet3=cluster_facet3
        pass

class MultiFacetDataset(Dataset):
    def __init__(self,data_file,cluster_file_facet1,cluster_file_facet2,cluster_file_facet3,debug) -> None:
        super(MultiFacetDataset,self).__init__()
        self.examples=[]
        f_text=open(data_file,encoding='utf-8')
        f1=open(cluster_file_facet1,encoding='utf-8')
        f2=open(cluster_file_facet2,encoding='utf-8')
        f3=open(cluster_file_facet3,encoding='utf-8')

        self.facet1_cluster=dict()
        self.facet2_cluster=dict()
        self.facet3_cluster=dict()
        self.cache=dict()
        self.corpus_set=[]
        
        for line_text,line1,line2,line3 in zip(f_text.readlines(),f1.readlines(),f2.readlines(),f3.readlines()):
            record=json.loads(line_text)
            record_facet1=json.loads(line1)
            record_facet2=json.loads(line2)
            record_facet3=json.loads(line3)
            for text, facet1_id, facet2_id, facet3_id in zip(record['context'],record_facet1['context_cluster'],record_facet2['context_cluster'],record_facet3['context_cluster']):
                try:
                    self.facet1_cluster[facet1_id].append(text)
                except:
                    self.facet1_cluster[facet1_id]=[text]
                try:
                    self.facet2_cluster[facet2_id].append(text)
                except:
                    self.facet2_cluster[facet2_id]=[text]
                try:
                    self.facet3_cluster[facet3_id].append(text)
                except:
                    self.facet3_cluster[facet3_id]=[text]
            self.examples.append(MultiFacetExample(dialog=record['context']+record['candidate'],cluster_facet1=record_facet1['context_cluster']+record_facet1['candidate_cluster'],cluster_facet2=record_facet2['context_cluster']+record_facet2['candidate_cluster'],cluster_facet3=record_facet3['context_cluster']+record_facet3['candidate_cluster']))
            self.corpus_set.extend(record['context'])
            if debug and len(self.examples)>=128:
                break
        
        self.corpus_set=set(self.corpus_set)
        # print(len(self.facet1_cluster.keys()))
        # print(len(self.facet2_cluster.keys()))
        # print(len(self.facet3_cluster.keys()))
        
    
    def __getitem__(self, index):
        example=self.examples[index]
        return example.dialog,example.cluster_facet1,example.cluster_facet2,example.cluster_facet3


    def __len__(self):
        return len(self.examples)

    def prepare_hard_candidate(self,cluster,n_sample=0):
        '''
        cluster: (bs,3,n_try)
        '''
        #(bs,n_sample)
        candidate_list=[]
        #(bs)
        facet1_idx=[]
        facet2_idx=[]
        facet3_idx=[]
        bs,n_try=len(cluster),len(cluster[0][0])
        for i in range(bs):
            for k1,k2,k3 in product(range(n_try),range(n_try),range(n_try)):
                if (k1,k2,k3) in self.cache.keys():
                    if len(self.cache[(k1,k2,k3)]):
                        facet1=cluster[i][0][k1]
                        facet2=cluster[i][1][k2]
                        facet3=cluster[i][2][k3]
                        sampled= random.choices(self.cache[(k1,k2,k3)],k=n_sample)
                    else:
                        sampled=None
                else:
                    facet1=cluster[i][0][k1]
                    facet2=cluster[i][1][k2]
                    facet3=cluster[i][2][k3]
                    if facet1 not in self.facet1_cluster.keys() or facet2 not in self.facet2_cluster.keys() or facet3 not in self.facet3_cluster.keys():
                        self.cache[(k1,k2,k3)]=[]
                        sampled=None
                    else:
                        intersection = set(self.facet1_cluster[facet1]).intersection(set(self.facet2_cluster[facet2])).intersection(set(self.facet3_cluster[facet3]))
                        if len(intersection):
                            self.cache[(k1,k2,k3)]=list(intersection)
                            sampled = random.choices(list(intersection),k=n_sample)
                        else:
                            self.cache[(k1,k2,k3)]=[]
                            sampled=None
                if sampled:
                    break
            if sampled is None:
                sampled=random.choices(list(self.facet3_cluster[0]),k=n_sample)
                facet1=random.choice(list(self.facet1_cluster.keys()))
                facet2=random.choice(list(self.facet2_cluster.keys()))
                facet3=0
                # raise NotImplementedError
            facet1_idx.append(facet1)
            facet2_idx.append(facet2)
            facet3_idx.append(facet3)
            candidate_list.append(sampled)
        assert len(candidate_list) == len(facet1_idx) == len(facet2_idx) == len(facet3_idx) == bs

        return candidate_list,facet1_idx,facet2_idx,facet3_idx


    def prepare_rand_candidate(self,bs,n_rand):
        candidate_list=[]
        for i in range(bs):
            sampled=random.choices(list(self.corpus_set),k=n_rand)
            candidate_list.append(sampled)
        assert len(candidate_list)==bs
        return candidate_list


    @staticmethod
    def collate_fn(batch):
        dialog_list=[item[0] for item in batch]
        facet1_list=[item[1] for item in batch]
        facet2_list=[item[2] for item in batch]
        facet3_list=[item[3] for item in batch]

        return dialog_list,facet1_list,facet2_list,facet3_list