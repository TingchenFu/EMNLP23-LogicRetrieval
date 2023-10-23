import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import random
from tqdm import tqdm
from str2bool import str2bool
import itertools
from datetime import datetime
from transformers import BertTokenizer
from transformers import BertConfig
from transformers import AdamW
from model.retrieval_model import CrossModel
from model.scheduling_model import Scheduler
from model.transition_model import Transition

from transformers.optimization import get_linear_schedule_with_warmup
from transformers.optimization import (
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from torch.utils.data import DataLoader, Dataset
from util import update_argument

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Pre-training for Knowledge-Grounded Conversation')
parser.add_argument("--debug",default=True,type=str2bool,help='debug mode, using small dataset')
parser.add_argument("--predict",type=str2bool,default=False)
parser.add_argument("--dataset",default='Douban',type=str,choices=['Douban','Ubuntu','Ecommerce'])
parser.add_argument("--train_file",type=str,default='path to the train file')
parser.add_argument("--eval_file",type=str,default='path to the eval file')
parser.add_argument("--train_facet1_file",type=str,help='train facet file')
parser.add_argument("--train_facet2_file",type=str,help='train facet file')
parser.add_argument("--train_facet3_file",type=str,help='train facet file')
parser.add_argument("--eval_facet1_file",type=str,help='evaluation facet file')
parser.add_argument("--eval_facet1_file",type=str,help='evaluation facet file')
parser.add_argument("--eval_facet1_file",type=str,help='evaluation facet file')



# training scheme
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--accum_step', type=int, default=4)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--clip', type=float, default=2.0)
parser.add_argument('--schedule', type=str, default='cosine')
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--warmup_step', type=int, default=500)
parser.add_argument('--n_step', type=int, default=1000000)
parser.add_argument('--n_epoch', type=int, default=3)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--eval_every', type=int, default=2000)
#parser.add_argument('--update_every',type=int,default=10000)
#parser.add_argument("--patience",type=int,default=3)
#parser.add_argument("--earlystop",type=float,default=0)
#parser.add_argument("--eval_mode",type=str,default='self')

# save 
parser.add_argument("--output_dir",type=str,help='path to the dump model checkpoints and training logs')
parser.add_argument('--log', type=str, default='log')
parser.add_argument('--seed', type=int, default=0)

# input
parser.add_argument("--max_utterance_length",type=int,default=32)
parser.add_argument("--max_context_length",type=int,default=256)
parser.add_argument("--max_cross_length",type=int,default=256)
parser.add_argument("--max_turn",type=int,default=15)

# vocab_path={
#     'ubuntu':'/home/tingchen_fu/PLM/MiniLM',
#     'douban':'/home/tingchen_fu/PLM/bert-base-chinese',
#     'ecommerce':'/home/tingchen_fu/PLM/bert-base-chinese'
# }

# model architecture
parser.add_argument("--n_facet1",type=int,default=256)
parser.add_argument("--n_facet2",type=int,default=64)
parser.add_argument("--n_facet3",type=int,default=16)
parser.add_argument("--hidden_dim",default=768)
parser.add_argument("--z_dim",type=int,default=128)

args = parser.parse_args()
args = update_argument(args,'warmup')
torch.cuda.empty_cache()
os.makedirs(args.output_dir,exist_ok=True)
with open(os.path.join(args.output_dir,'args.json'),'w') as f:
    json.dump(vars(args),f)

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# args.device=torch.device('cpu')
if args.debug:
    args.batch_size=2
    args.eval_batch_size=2
    args.print_every=1
    args.eval_every=2


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark=True

logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "log"), 'w'))

# Build dataset
logger.info("Create dataset begin...  ")
from dataset import MultiFacetDataset
from batcher import MultiFacetBatcher
eval_dataset=MultiFacetDataset(args.eval_file,args.eval_facet1_file,args.eval_facet2_file,args.eval_facet3_file,args.debug)
eval_loader=DataLoader(eval_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=MultiFacetDataset.collate_fn)

train_dataset=MultiFacetDataset(args.train_file,args.train_facet1_file,args.train_facet2_file,args.train_facet3_file,args.debug)
train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=not args.debug,collate_fn=MultiFacetDataset.collate_fn)
train_loader=itertools.cycle(train_loader)
time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("Create dataset end... | %s " % time_str)

logger.info('Training Dataset {}'.format(len(train_dataset)))
logger.info('Eval Dataset {}'.format(len(eval_dataset)))
tokenizer=BertTokenizer.from_pretrained(args.vocab_path)
tokenizer.add_special_tokens({'pad_token':'[PAD]','unk_token':'[UNK]','eos_token':'[EOS]'})
batcher = MultiFacetBatcher(tokenizer,args.max_turn,args.max_utterance_length,args.device)
configuration=BertConfig.from_json_file(args.config_path)
configuration.num_hidden_layers=3
transition_model=Transition(configuration,args.hidden_dim,args.z_dim)

transition_model.inference_transformer.resize_token_embeddings(len(tokenizer))
transition_model.generative_transformer.resize_token_embeddings(len(tokenizer))

transition_model.to(args.device)


no_decay = ["bias", "LayerNorm.weight"]
grouped_parameters = [
    {
        "params": [p for n, p in transition_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in transition_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
if not args.predict:
    total_steps = args.n_epoch * (len(train_dataset) / (args.batch_size * args.accum_step))
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    if args.schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_step, num_training_steps=total_steps)
    elif args.schedule == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_step, num_training_steps=total_steps)

def train_step(global_step):
    transition_model.train()
    for _ in range(args.accum_step):
        dialog_list,facet1_list,facet2_list,facet3_list = next(train_loader)
        batch_dict=batcher.preprocess(dialog_list,facet1_list,facet2_list,facet3_list)
        #(bs,n_turn,seq_len)
        dialog_id=batch_dict['context_id']
        facet1_id=batch_dict['context_facet1_id']
        facet2_id=batch_dict['context_facet2_id']
        facet3_id=batch_dict['context_facet3_id']
        bs,n_turn=dialog_id.shape[0],dialog_id.shape[1]
        #(bs,n_turn)
        turn_mask=torch.any(dialog_id!=tokenizer.pad_token_id,dim=-1)
        
        post_loss=0.0
        prior_loss=0.0
        for turn in range(n_turn):
            #(bs,seq_len)
            utterance_id=dialog_id[:,turn,:]
            utterance_mask=(utterance_id!=tokenizer.pad_token_id)
            post_pis,post_mus,post_sigmas=transition_model.inference(utterance_id,utterance_mask)
            
            post_loss+=F.cross_entropy(post_pis[0],facet1_id[:,turn])
            post_loss+=F.cross_entropy(post_pis[1],facet2_id[:,turn])
            post_loss+=F.cross_entropy(post_pis[2],facet3_id[:,turn])
            
        # post_loss=post_loss/turn_mask.sum(1).sum(0)
        

        #(bs,n_turn,n_cluster), (bs,n_turn,zdim), (bs,n_turn,zdim)
        prior_pis,prior_mus,prior_sigmas=transition_model.prior(facet1_id,facet2_id,facet3_id,turn_mask)
        prior_pi1,prior_pi2,prior_pi3=prior_pis[0],prior_pis[1],prior_pis[2]
        
        # print(prior_pi1.shape)
        # print(prior_pi2.shape)
        # print(prior_pi3.shape)
        prior_loss += F.cross_entropy(prior_pi1[:,:-1,:].contiguous().view(-1, args.n_facet1), facet1_id[:,1:].contiguous().view(-1))
        prior_loss += F.cross_entropy(prior_pi2[:,:-1,:].contiguous().view(-1, args.n_facet2), facet2_id[:,1:].contiguous().view(-1))
        prior_loss += F.cross_entropy(prior_pi3[:,:-1,:].contiguous().view(-1, args.n_facet3), facet3_id[:,1:].contiguous().view(-1))

        loss=prior_loss+post_loss

        loss=loss/args.accum_step
        loss.backward()

    grad_norm_cross = torch.nn.utils.clip_grad_norm_([p for p in transition_model.parameters() if p.requires_grad], args.clip)
    if grad_norm_cross >= 1e2:
        logger.info('WARNING : Exploding Poly Gradients {:.2f}'.format(grad_norm_cross))
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if global_step % args.print_every == 0 and global_step != 0:
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info("Step: %d \t| prior_loss: %.3f \t | post_loss: %.3f \t   | lr: %.8f \t| %s" % (
            global_step, prior_loss, post_loss, scheduler.get_lr()[0], time_str
        ))

def predict_step(global_step,best_metric):
    transition_model.eval()
    post_hit1=0.0
    post_hit2=0.0
    post_hit3=0.0
    prior_hit1=0.0
    prior_hit2=0.0
    prior_hit3=0.0
    count_post=0
    count_prior=0
    for dialog_list,facet1_list,facet2_list,facet3_list in eval_loader:
        batch_dict=batcher.preprocess(dialog_list,facet1_list,facet2_list,facet3_list)
        #(bs,n_turn,seq_len)
        dialog_id=batch_dict['context_id']
        facet1_id=batch_dict['context_facet1_id']
        facet2_id=batch_dict['context_facet2_id']
        facet3_id=batch_dict['context_facet3_id']
        bs,n_turn=dialog_id.shape[0],dialog_id.shape[1]
        count_prior+=bs
        #(bs,n_turn)
        turn_mask=torch.any(dialog_id!=tokenizer.pad_token_id,dim=-1)
        count_post += turn_mask.sum(1).sum(0).item()

        with torch.no_grad():
            for turn in range(n_turn):
                #(bs,seq_len)
                utterance_id=dialog_id[:,turn,:]
                utterance_mask=(utterance_id!=tokenizer.pad_token_id)
                post_pis,post_mus,post_sigmas=transition_model.inference(utterance_id,utterance_mask) 
                
                post_hit1+=(torch.argmax(post_pis[0],dim=1) == facet1_id[:,turn]).to(torch.uint8).sum(0).item()
                post_hit2+=(torch.argmax(post_pis[1],dim=1) == facet2_id[:,turn]).to(torch.uint8).sum(0).item()
                post_hit3+=(torch.argmax(post_pis[2],dim=1) == facet3_id[:,turn]).to(torch.uint8).sum(0).item()

            prior_pis,prior_mus,prior_sigmas=transition_model.prior(facet1_id,facet2_id,facet3_id,turn_mask)
            response_pi1=torch.gather(prior_pis[0],dim=1,index=(turn_mask.sum(1,keepdim=True)-1).unsqueeze(1).repeat(1,1,args.n_facet1)).squeeze(1)
            response_pi2=torch.gather(prior_pis[1],dim=1,index=(turn_mask.sum(1,keepdim=True)-1).unsqueeze(1).repeat(1,1,args.n_facet2)).squeeze(1)
            response_pi3=torch.gather(prior_pis[2],dim=1,index=(turn_mask.sum(1,keepdim=True)-1).unsqueeze(1).repeat(1,1,args.n_facet3)).squeeze(1)


            prior_hit1+= (torch.argmax(response_pi1,dim=1) == batch_dict['candidate_facet1_id'][:,0]).to(torch.uint8).sum(0).item()
            prior_hit2+= (torch.argmax(response_pi2,dim=1) == batch_dict['candidate_facet2_id'][:,0]).to(torch.uint8).sum(0).item()
            prior_hit3+= (torch.argmax(response_pi3,dim=1) == batch_dict['candidate_facet3_id'][:,0]).to(torch.uint8).sum(0).item()
    
    current_metric = post_hit1/count_post + post_hit2/count_post + post_hit3/count_post + prior_hit1/count_prior + prior_hit2/count_prior + prior_hit3/count_prior
    logger.info("**********************************")
    logger.info("test results..........")
    logger.info("post precision {} {} {} ".format(post_hit1/count_post, post_hit2/count_post, post_hit3/count_post))
    logger.info("prior precision {} {} {} ".format(prior_hit1/count_prior, prior_hit2/count_prior, prior_hit3/count_prior))
    logger.info("current_metric:{}".format(current_metric))

    if not args.predict and current_metric >  best_metric:
        cross_save_path=os.path.join(args.output_dir,'{}step_transition_model'.format(global_step))
        torch.save(transition_model.state_dict(),cross_save_path)

    return max(current_metric, best_metric)
        

def encode_step(global_step):
    transition_model.eval()
    y_facet1=[]
    y_facet2=[]
    y_facet3=[]
    turns=[]
    for dialog_list,facet1_list,facet2_list,facet3_list in eval_loader:
        batch_dict=batcher.encode(dialog_list)
        #(bs,seq_len)
        corpus_id=batch_dict['corpus_id']
        turns.extend([len(x)] for x in dialog_list)
        with torch.no_grad():
            #(bs,seq_len)
            utterance_id=corpus_id
            utterance_mask=(corpus_id!=tokenizer.pad_token_id)
            post_pis,post_mus,post_sigmas=transition_model.inference(utterance_id,utterance_mask)
            
            y_facet1.extend(post_pis[0].argmax(dim=1).detech().cpu().tolist())
            y_facet2.extend(post_pis[1].argmax(dim=1).detech().cpu().tolist())
            y_facet3.extend(post_pis[2].argmax(dim=1).detech().cpu().tolist())

    f=open(os.path.join(args.output_dir,'{}step_facet1_cluster'),'w')
    offset=0
    for i in range(len(turns)):
        session_facet1=y_facet1[offset:offset+turns[i]]
        f.write(json.dumps({'context_cluster':session_facet1[:-2],'candidate_cluster':session_facet1[-2:]},ensure_ascii=False)+'\n')
        offset+=turns[i]
    f.close()

    f=open(os.path.join(args.output_dir,'{}step_facet2_cluster'),'w')
    offset=0
    for i in range(len(turns)):
        session_facet2=y_facet2[offset:offset+turns[i]]
        f.write(json.dumps({'context_cluster':session_facet2[:-2],'candidate_cluster':session_facet2[-2:]},ensure_ascii=False)+'\n')
        offset+=turns[i]
    f.close()

    f=open(os.path.join(args.output_dir,'{}step_facet3_cluster'),'w')
    offset=0
    for i in range(len(turns)):
        session_facet3=y_facet3[offset:offset+turns[i]]
        f.write(json.dumps({'context_cluster':session_facet3[:-2],'candidate_cluster':session_facet3[-2:]},ensure_ascii=False)+'\n')
        offset+=turns[i]
    f.close()


best_metric = -1.0
# if args.predict:
#     if args.eval_mode=='official':
#         raise NotImplementedError
#     elif args.eval_mode=='self':
#         self_predict_step(0,best_metric)
#     #logger.info("predict result: the f1 between predict knowledge and response: {:.6f}".format(f1))
#     exit()
for i in range(args.n_step):
    train_step(i + 1)
    if (i+1)%args.eval_every==0:
        best_metric=predict_step(i+1,best_metric)