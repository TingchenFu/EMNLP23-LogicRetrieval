import argparse
import os
import json
#os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ['CUDA_VISIBLE_DEVICES']='1'
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
parser.add_argument("--debug",default=False,type=str2bool,help='debug mode, using small dataset')
parser.add_argument("--predict",type=str2bool,default=False)
parser.add_argument("--encode",type=str2bool,default=True)
parser.add_argument("--dataset",default='Douban',type=str,choices=['Douban','Ubuntu'])

# cluster_file={
#     'ubuntu':'/home/tingchen_fu/UbuntuV1/c3dist/cluster_result.jsonl',
#     'douban':'/home/tingchen_fu/DialogStructure/dump_douban/cluster/cluster_result.jsonl',
#     'ecommerce':'/home/tingchen_fu/DialogStructure/dump_ecommerce/cluster/cluster_result.jsonl',
# }

# training scheme
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--eval_batch_size', type=int, default=128)
parser.add_argument('--accum_step', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--clip', type=float, default=2.0)
parser.add_argument('--schedule', type=str, default='cosine')
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--warmup_step', type=int, default=500)
parser.add_argument('--n_step', type=int, default=1000000)
parser.add_argument('--n_epoch', type=int, default=3)
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--eval_every', type=int, default=5000)
#parser.add_argument('--update_every',type=int,default=10000)
#parser.add_argument("--patience",type=int,default=3)
#parser.add_argument("--earlystop",type=float,default=0)
#parser.add_argument("--eval_mode",type=str,default='self')

parser.add_argument("--transition_model_path",type=str,default='/home/zhaoxl/DialogueStructure/dump/warmup_Douban/bs32lr5.0warm500_Feb1223:57/20000step_transition_model')

# save 
parser.add_argument("--dump_path",type=str,default='/home/zhaoxl/DialogueStructure/dump')
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
args = update_argument(args,'vae')
torch.cuda.empty_cache()
os.makedirs(args.output_dir,exist_ok=True)
with open(os.path.join(args.output_dir,'args.json'),'w') as f:
    json.dump(vars(args),f)

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#args.device=torch.device('cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark=True

logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "log"), 'w'))

# Build dataset
logger.info("Create dataset begin...  ")
from dataset import MultiFacetDataset
from batcher import MultiFacetBatcher

train_dataset=MultiFacetDataset(args.train_file,args.train_facet1_file,args.train_facet2_file,args.train_facet3_file,args.debug)
single_train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=not args.debug,collate_fn=MultiFacetDataset.collate_fn)
train_loader=itertools.cycle(single_train_loader)
encode_loader=DataLoader(train_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=MultiFacetDataset.collate_fn)


eval_dataset=MultiFacetDataset(args.eval_file,args.eval_facet1_file, args.eval_facet2_file,args.eval_facet3_file, args.debug)
eval_loader=DataLoader(eval_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=MultiFacetDataset.collate_fn)
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

reloaded=torch.load(args.transition_model_path)
transition_model.load_state_dict(reloaded,strict=True)


transition_model.to(args.device)
del reloaded

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
    step_KLloss=0.0
    step_recloss=0.0
    for _ in range(args.accum_step):
        dialog_list,facet1_list,facet2_list,facet3_list = next(train_loader)
        batch_dict=batcher.preprocess(dialog_list,facet1_list,facet2_list,facet3_list)
        #(bs,n_turn,seq_len)
        dialog_id=batch_dict['input_id']
        facet1_id=batch_dict['facet1_id']
        facet2_id=batch_dict['facet2_id']
        facet3_id=batch_dict['facet3_id']
        bs,n_turn=dialog_id.shape[0],dialog_id.shape[1]
        #(bs,n_turn)
        turn_mask=torch.any(dialog_id!=tokenizer.pad_token_id,dim=-1)
        
        post_pi1=[]
        post_pi2=[]
        post_pi3=[]
        post_mu1=[]
        post_mu2=[]
        post_mu3=[]
        post_sigma1=[]
        post_sigma2=[]
        post_sigma3=[]
        
        reconstruction_loss=0.0
        for turn in range(n_turn):
            #(bs,seq_len)
            utterance_id=dialog_id[:,turn,:]
            utterance_mask=(utterance_id!=tokenizer.pad_token_id)
            post_pis,post_mus,post_sigmas=transition_model.inference(utterance_id,utterance_mask)
            #(bs,n_cluster)
            post_pi1.append(post_pis[0])
            post_pi2.append(post_pis[1])
            post_pi3.append(post_pis[2])
            #(bs,zdim)
            post_mu1.append(post_mus[0])
            post_mu2.append(post_mus[1])
            post_mu3.append(post_mus[2])
            post_sigma1.append(post_sigmas[0])
            post_sigma2.append(post_sigmas[1])
            post_sigma3.append(post_sigmas[2])
            reconstruction_loss += transition_model.generate(utterance_id,utterance_mask,post_mus[0]+post_sigmas[0]*np.random.normal(),post_mus[1]+post_sigmas[1]*np.random.normal(),post_mus[2]+post_sigmas[2]*np.random.normal())
        reconstruction_loss=reconstruction_loss/((dialog_id!=tokenizer.pad_token_id).sum(1).sum(0)) 
        
        #(bs,n_turn,n_cluster)
        post_pi1=torch.stack(post_pi1,dim=1)
        post_pi2=torch.stack(post_pi2,dim=1)
        post_pi3=torch.stack(post_pi3,dim=1)

        #(bs, n_turn, zdim)
        post_mu1=torch.stack(post_mu1,dim=1)
        post_mu2=torch.stack(post_mu2,dim=1)
        post_mu3=torch.stack(post_mu3,dim=1)

        #(bs, n_turn, zdim)
        post_sigma1=torch.stack(post_sigma1,dim=1)
        post_sigma2=torch.stack(post_sigma2,dim=1)
        post_sigma3=torch.stack(post_sigma3,dim=1)

        z_dim=post_mu1.shape[2]

        #(bs,n_turn,n_cluster), (bs,n_turn,zdim), (bs,n_turn,zdim)
        prior_pis,prior_mus,prior_sigmas=transition_model.prior(post_pi1,post_pi2,post_pi3)
        prior_pi1,prior_pi2,prior_pi3=prior_pis[0],prior_pis[1],prior_pis[2]
        prior_mu1,prior_mu2,prior_mu3=prior_mus[0],prior_mus[1],prior_mus[2]
        prior_sigma1,prior_sigma2,prior_sigma3=prior_sigmas[0],prior_sigmas[1],prior_sigmas[2]
        
        #(bs,n_turn)
        KLy1=F.kl_div(prior_pi1, post_pi1.log(),reduction='none')
        KLy2=F.kl_div(prior_pi2, post_pi2.log(),reduction='none')
        KLy3=F.kl_div(prior_pi3, post_pi3.log(),reduction='none')

        KLy1=(KLy1*turn_mask)/turn_mask.sum(2).sum(1).item()
        KLy2=(KLy2*turn_mask)/turn_mask.sum(2).sum(1).item()
        KLy3=(KLy3*turn_mask)/turn_mask.sum(2).sum(1).item()

        KLz1 = (post_sigma1/prior_sigma1).sum(2) + ((prior_mu1 - post_mu1) * (prior_mu1-post_mu1) / prior_sigma1).sum(2) - z_dim + torch.log(prior_sigma1.prod(2)/ post_sigma1.prod(2))
        KLz2 = (post_sigma2/prior_sigma2).sum(2) + ((prior_mu2 - post_mu2) * (prior_mu2-post_mu2) / prior_sigma2).sum(2) - z_dim + torch.log(prior_sigma2.prod(2)/ post_sigma2.prod(2))
        KLz3 = (post_sigma3/prior_sigma3).sum(2) + ((prior_mu3 - post_mu3) * (prior_mu3-post_mu3) / prior_sigma3).sum(2) - z_dim + torch.log(prior_sigma3.prod(2)/ post_sigma3.prod(2))

        KLz1=(KLz1*turn_mask)/turn_mask.sum(2).sum(1).item()
        KLz2=(KLz2*turn_mask)/turn_mask.sum(2).sum(1).item()
        KLz3=(KLz3*turn_mask)/turn_mask.sum(2).sum(1).item()

        loss=reconstruction_loss+KLy1+KLy2+KLy3+KLz1+KLz2+KLz3
        
        step_KLloss+=KLy1.item()+KLy2.item()+KLy3.item()
        step_recloss+=reconstruction_loss.item()
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
        logger.info("Step: %d \t| rs_loss: %.3f \t | kl_loss: %.3f \t   | lr: %.8f \t| %s" % (
            global_step, reconstruction_loss, scheduler.get_lr()[0], time_str
        ))

def encode_step(global_step):
    transition_model.eval()
    y_facet1=[]
    y_facet2=[]
    y_facet3=[]
    turns=[]
    for dialog_list,facet1_list,facet2_list,facet3_list in tqdm(encode_loader):
        turns.extend([len(x) for x in dialog_list])
        batch_dict=batcher.encode(dialog_list)
        #(bs,seq_len)
        corpus_id=batch_dict['corpus_id']
        with torch.no_grad():
            #(bs,seq_len)
            utterance_id=corpus_id
            utterance_mask=(corpus_id!=tokenizer.pad_token_id)
            pi1,pi2,pi3=transition_model.encode(utterance_id,utterance_mask)
            
            y_facet1.extend(pi1.argmax(dim=1).detach().cpu().tolist())
            y_facet2.extend(pi2.argmax(dim=1).detach().cpu().tolist())
            y_facet3.extend(pi3.argmax(dim=1).detach().cpu().tolist())

    assert len(y_facet1) == sum(turns)
    f=open(os.path.join(args.output_dir,'{}stepfacet1_cluster'.format(global_step)),'w')
    offset=0
    for i in range(len(turns)):
        session_facet1=y_facet1[offset:offset+turns[i]]
        f.write(json.dumps({'context_cluster':session_facet1[:-2],'candidate_cluster':session_facet1[-2:]},ensure_ascii=False)+'\n')
        offset+=turns[i]
    f.close()

    f=open(os.path.join(args.output_dir,'{}stepfacet2_cluster'.format(global_step)),'w')
    offset=0
    for i in range(len(turns)):
        session_facet2=y_facet2[offset:offset+turns[i]]
        f.write(json.dumps({'context_cluster':session_facet2[:-2],'candidate_cluster':session_facet2[-2:]},ensure_ascii=False)+'\n')
        offset+=turns[i]
    f.close()

    f=open(os.path.join(args.output_dir,'{}stepfacet3_cluster'.format(global_step)),'w')
    offset=0
    for i in range(len(turns)):
        session_facet3=y_facet3[offset:offset+turns[i]]
        f.write(json.dumps({'context_cluster':session_facet3[:-2],'candidate_cluster':session_facet3[-2:]},ensure_ascii=False)+'\n')
        offset+=turns[i]
    f.close()


best_metric = -1.0
if args.encode:
    encode_step(0)
    exit()
for i in range(args.n_step):
    train_step(i + 1)
