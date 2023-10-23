'''
  @Date    : 2021-12-20
  @Author  : Tingchen Fu
  @Description: 
  the ensemble of two components: M (cluster classifer) and the discriminator, we remove the G2(generator component) here 
  The M12 is not appear in the code but the cluster results from M12
'''
import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
#os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import json
import random
from tqdm import tqdm
from str2bool import str2bool
import itertools
from datetime import datetime
from transformers import BertTokenizer
from transformers import BertConfig
from transformers import AdamW
from model.scheduling_model import Scheduler
from model.retrieval_model import CrossModelv2
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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Pre-training for Knowledge-Grounded Conversation')
parser.add_argument("--debug",default=True,type=str2bool,help='debug mode, using small dataset')
parser.add_argument("--predict",type=str2bool,default=False)
parser.add_argument("--dataset",default='Douban',type=str,choices=['Douban','Ubuntu'])
parser.add_argument("--train_file",type=str,help='path to the training set')
parser.add_argument("--eval_file",type=str,help='path to the evaluation file')
parser.add_argument("--facet1_file", type=str,help="path to the first facet file")
parser.add_argument("--facet2_file", type=str,help="path to the second facet file")
parser.add_argument("--facet3_file", type=str,help="path to the third facet file")
parser.add_argument("--base",default='bert',type=str,choices=['bert','ums','fp','sabert'])

# # sampling
parser.add_argument("--n_rand",type=int,default=1)
parser.add_argument("--n_inter",type=int,default=1)
parser.add_argument("--n_transition",type=int,default=1)
parser.add_argument("--n_try",type=int,default=3,help="the number of samples from scheudling network")
parser.add_argument("--n_sample",type=int,default=0,help="the number of samples from scheudling network to compute the reward")
parser.add_argument("--keep",help='whether keep the original negative in the training set',type=str2bool,default=True)
parser.add_argument("--naive",type=str2bool,default=True)

# training scheme
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--n_step', type=int, default=1000000)
parser.add_argument('--accum_step', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--clip', type=float, default=2.0)
parser.add_argument('--schedule', type=str, default='cosine')
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--warmup_step', type=int, default=500)
parser.add_argument('--n_epoch', type=int, default=3)
parser.add_argument("--output_dir",type=str,help='the path to output model checkpoint and log file')
parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--eval_every', type=int, default=2500)

# input
parser.add_argument("--max_utterance_length",type=int,default=32)
parser.add_argument("--max_context_length",type=int,default=256)
parser.add_argument("--max_cross_length",type=int,default=256)
parser.add_argument("--max_turn",type=int,default=15)

parser.add_argument("--transition_model_path",type=str,default=None)
parser.add_argument("--transition_args",type=str,default=None)
parser.add_argument("--cross_model_path",type=str,default=None)

parser.add_argument("--n_layer",type=int,default=1,help='the layers in scheduling network')


args = parser.parse_args()
# from util import update_argument
# args=update_argument(args,'interaction')

if args.debug:
    args.print_every=1
    args.eval_every=8
    args.batch_size=2
    args.eval_batch_size=2

os.makedirs(args.output_dir,exist_ok=True)
logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "log"), 'w'))
f=open(os.path.join(args.output_dir,'args.json'),'w',encoding='utf-8')
json.dump(vars(args),f)
logger.info("\nParameters:")
for attr, value in sorted(vars(args).items()):
    logger.info("{}={}".format(attr.upper(), value))


torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark=True

from batcher import MultiFacetBatcher
from dataset import MultiFacetDataset,InferenceDataset

# Build dataset
time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("Create dataset begin... | %s " % time_str)

train_dataset = MultiFacetDataset(args.train_file,args.facet1_file,args.facet2_file,args.facet3_file,args.debug)
train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=not args.debug,collate_fn=MultiFacetDataset.collate_fn)
train_loader = itertools.cycle(train_loader)

eval_dataset = InferenceDataset(args.eval_file,  args.debug)
eval_loader = DataLoader(eval_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=InferenceDataset.collate_fn)


time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("Create dataset end... | %s " % time_str)
if not args.predict:
    logger.info('Training Dataset {}'.format(len(train_dataset)))
logger.info('Eval Dataset {}'.format(len(eval_dataset)))
tokenizer=BertTokenizer.from_pretrained(args.vocab_path)
tokenizer.add_special_tokens({'pad_token':'[PAD]','unk_token':'[UNK]','eos_token':'[EOS]'})
batcher = MultiFacetBatcher(tokenizer,args.max_turn,args.max_utterance_length,device)

configuration = BertConfig.from_json_file(args.config_path)
cross_model = CrossModelv2(configuration)
configuration.num_hidden_layers=3
transition_args = json.load(open(args.transition_args))
transition_model = Transition(configuration,hidden_dim=transition_args['hidden_dim'],z_dim=transition_args['z_dim'])
configuration.num_hidden_layers = args.n_layer
scheduling_model=Scheduler(transition_args['n_facet1'], transition_args['n_facet2'], transition_args['n_facet3'])
configuration.num_hidden_layers = 3

if args.transition_model_path:
    reloaded=torch.load(args.transition_model_path, map_location='cpu')
    transition_model.inference_transformer.resize_token_embeddings(len(tokenizer))
    transition_model.generative_transformer.resize_token_embeddings(len(tokenizer))
    transition_model.load_state_dict(reloaded,strict=True)
    transition_model.to(device)
    del reloaded

scheduling_model.to(device)

reloaded=torch.load(args.cross_model_path)
cross_model.load_state_dict(reloaded,strict=False)
cross_model.bert.resize_token_embeddings(len(tokenizer))
cross_model.to(device)
del reloaded

no_decay = ["bias", "LayerNorm.weight"]
cross_grouped_parameters = [
    {
        "params": [p for n, p in cross_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in cross_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
scheduling_grouped_parameters = [
    {
        "params": [p for n, p in scheduling_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in scheduling_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

cross_optimizer = AdamW(cross_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
scheduling_optimizer = AdamW(scheduling_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
total_steps = args.n_epoch * (len(train_dataset) / (args.batch_size * args.accum_step))
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
if args.schedule == 'linear':
    cross_scheduler = get_linear_schedule_with_warmup(cross_optimizer, num_warmup_steps=args.warmup_step, num_training_steps=total_steps)
    scheduling_scheduler = get_linear_schedule_with_warmup(scheduling_optimizer, num_warmup_steps=args.warmup_step, num_training_steps=total_steps)
elif args.schedule == 'cosine':
    cross_scheduler = get_cosine_schedule_with_warmup(cross_optimizer, num_warmup_steps=args.warmup_step, num_training_steps=total_steps)
    scheduling_scheduler = get_cosine_schedule_with_warmup(scheduling_optimizer, num_warmup_steps=args.warmup_step, num_training_steps=total_steps)

def scheduling_step(global_step):
    step_scheduling_loss=0.0
    for _ in range(args.accum_step):
        dialog_list, facet1_list, facet2_list, facet3_list = next(train_loader)

        # first part: train scheduling mode
        transition_model.eval()
        cross_model.eval()
        scheduling_model.train()

        bs=len(dialog_list)
        batch_dict = batcher.preprocess(dialog_list,facet1_list,facet2_list,facet3_list)
        context_id=batch_dict['context_id']
        #(bs,n_turn)
        turn_mask=torch.any(context_id!=tokenizer.pad_token_id,dim=-1)
        context_facet1_id=batch_dict['context_facet1_id']
        context_facet2_id=batch_dict['context_facet2_id']
        context_facet3_id=batch_dict['context_facet3_id']
        
        with torch.no_grad():
            pis,mus,sigmas = transition_model.prior(context_facet1_id,context_facet2_id,context_facet3_id)
            pi1,pi2,pi3=pis[0],pis[1],pis[2]
        pi1,pi2,pi3=scheduling_model(pi1,pi2,pi3)
        
        #(bs,n_cluster)
        response_pi1=torch.gather(pi1,dim=1,index=(turn_mask.sum(1,keepdim=True)-1).unsqueeze(-1).repeat(1,1,transition_args['n_facet1'])).squeeze(1)
        response_pi2=torch.gather(pi2,dim=1,index=(turn_mask.sum(1,keepdim=True)-1).unsqueeze(-1).repeat(1,1,transition_args['n_facet2'])).squeeze(1)
        response_pi3=torch.gather(pi3,dim=1,index=(turn_mask.sum(1,keepdim=True)-1).unsqueeze(-1).repeat(1,1,transition_args['n_facet3'])).squeeze(1)
        #(bs, n_cluster)
        response_pi1 = F.softmax(response_pi1,dim=1)
        response_pi2 = F.softmax(response_pi2,dim=1)
        response_pi3 = F.softmax(response_pi3,dim=1)
        
        #(bs,3,n_try)
        cluster=torch.stack([torch.multinomial(response_pi1,num_samples=args.n_try),torch.multinomial(response_pi2,num_samples=args.n_try),torch.multinomial(response_pi3,num_samples=args.n_try)],dim=1).detach().cpu().tolist()
        hard_candidate_list,facet1_index,facet2_index,facet3_index=train_dataset.prepare_hard_candidate(cluster,args.n_sample)
        
        facet1_index=torch.tensor(facet1_index,dtype=torch.long,device=device)
        facet2_index=torch.tensor(facet2_index,dtype=torch.long,device=device)
        facet3_index=torch.tensor(facet3_index,dtype=torch.long,device=device)
        #(bs)
        prob_facet1 = torch.gather(response_pi1,dim=1,index=facet1_index.unsqueeze(1)).squeeze(1)
        prob_facet2 = torch.gather(response_pi2,dim=1,index=facet2_index.unsqueeze(1)).squeeze(1)
        prob_facet3 = torch.gather(response_pi3,dim=1,index=facet3_index.unsqueeze(1)).squeeze(1)
        

        
        context_list=[x[:-2] for x in dialog_list]
        if args.n_rand:
            random_candidate_list = [random.choices( context_list[i-1],k = args.n_rand ) for i in range(bs)]
            #random_candidate_list = train_dataset.prepare_rand_candidate(bs,args.n_rand)
        else:
            random_candidate_list=[[] for i in range(bs)]

        if args.keep:
            candidate_list=[dialog_list[i][-2:]+ random_candidate_list[i] + hard_candidate_list[i] for i in range(bs)   ]
        else:
            candidate_list=[dialog_list[i][-2:-1]+ random_candidate_list[i] + hard_candidate_list[i] for i in range(bs)]
        
        if args.n_inter:
            candidate_list=[candidate_list[i]+random.choices(context_list[i],k=args.n_inter) for i in range(bs)]
        
        batch_dict=batcher.retrieve(context_list,candidate_list)
        input_id=batch_dict['input_id']
        segment_id=batch_dict['segment_id']

        with torch.no_grad():
            logit=cross_model(input_id,(input_id!=tokenizer.pad_token_id),segment_id)
            #(bs)
            reward=F.softmax(logit,dim=-1)[:,0].detach()
        
        # print(prob_facet1.device)
        # print(reward.device)
        loss=(prob_facet1+prob_facet2+prob_facet3) * reward
        loss=loss.sum(0)
        step_scheduling_loss+=loss.item()
        loss = loss / args.accum_step
        loss.backward()

    grad_norm_cross = torch.nn.utils.clip_grad_norm_([p for p in scheduling_model.parameters() if p.requires_grad], args.clip)
    if grad_norm_cross >= 1e2:
        logger.info('WARNING : Exploding Poly Gradients {:.2f}'.format(grad_norm_cross))
    scheduling_optimizer.step()
    scheduling_scheduler.step()
    scheduling_optimizer.zero_grad()

    if global_step % args.print_every == 0 and global_step != 0:
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info("Step: %d \t| scheduling_loss: %.3f \t| lr: %.8f \t| %s" % (
            global_step, step_scheduling_loss, cross_scheduler.get_lr()[0], time_str
        ))

def retrieval_step(global_step):
    # second part: train retrieval mode
    transition_model.eval()
    cross_model.train()
    scheduling_model.eval()
    step_retrieval_loss=0.0
    for _ in range(args.accum_step):
        dialog_list, facet1_list, facet2_list, facet3_list = next(train_loader)
        bs=len(dialog_list)
        batch_dict = batcher.preprocess(dialog_list,facet1_list,facet2_list,facet3_list)
        context_facet1_id=batch_dict['context_facet1_id']
        context_facet2_id=batch_dict['context_facet2_id']
        context_facet3_id=batch_dict['context_facet3_id']
        context_id=batch_dict['context_id']
        #(bs,n_turn)
        turn_mask=torch.any(context_id!=tokenizer.pad_token_id,dim=-1)
        with torch.no_grad():
            pis,mus,sigmas = transition_model.prior(context_facet1_id,context_facet2_id,context_facet3_id)
            # (bs,n_turn,n_facet) note the pi here is raw logit but not probability 
            pi1,pi2,pi3=pis[0],pis[1],pis[2]
            if not args.naive:
                pi1,pi2,pi3=scheduling_model(pi1,pi2,pi3)
            #(bs,n_cluster)
            response_pi1=torch.gather(pi1,dim=1,index=(turn_mask.sum(1,keepdim=True)-1).unsqueeze(-1).repeat(1,1,transition_args['n_facet1'])).squeeze(1)
            response_pi2=torch.gather(pi2,dim=1,index=(turn_mask.sum(1,keepdim=True)-1).unsqueeze(-1).repeat(1,1,transition_args['n_facet2'])).squeeze(1)
            response_pi3=torch.gather(pi3,dim=1,index=(turn_mask.sum(1,keepdim=True)-1).unsqueeze(-1).repeat(1,1,transition_args['n_facet3'])).squeeze(1)

            response_pi1 = F.softmax(response_pi1,dim=1)
            response_pi2 = F.softmax(response_pi2,dim=1)
            response_pi3 = F.softmax(response_pi3,dim=1)

            # (bs,n_try,3)
            cluster=torch.stack([torch.multinomial(response_pi1,num_samples=args.n_try),torch.multinomial(response_pi2,num_samples=args.n_try),torch.multinomial(response_pi3,num_samples=args.n_try)],dim=1).detach().cpu().tolist()

            #(bs,3,1)
            #cluster=torch.stack([batch_dict['candidate_facet1_id'][:,1:2], batch_dict['candidate_facet2_id'][:,1:2],batch_dict['candidate_facet3_id'][:,1:2]],dim=1).detach().cpu().tolist()        
            if args.n_transition:
                hard_candidate_list,facet1_index,facet2_index,facet3_index=train_dataset.prepare_hard_candidate(cluster,args.n_transition)
            else:
                hard_candidate_list=[[] for i in range(bs)]
        
        
        context_list=[x[:-2] for x in dialog_list]
        if args.n_rand:
            random_candidate_list = [random.choices( context_list[i-1],k = args.n_rand ) for i in range(bs)]
            #random_candidate_list = train_dataset.prepare_rand_candidate(bs,args.n_rand)
        else:
            random_candidate_list=[[] for i in range(bs)]

        if args.keep:
            candidate_list=[dialog_list[i][-2:]+ random_candidate_list[i] + hard_candidate_list[i] for i in range(bs)   ]
        else:
            candidate_list=[dialog_list[i][-2:-1]+ random_candidate_list[i] + hard_candidate_list[i] for i in range(bs)]
        
        if args.n_inter:
            candidate_list=[candidate_list[i]+random.choices(context_list[i],k=args.n_inter) for i in range(bs)]
        
        batch_dict=batcher.retrieve(context_list,candidate_list)
        input_id=batch_dict['input_id']
        segment_id=batch_dict['segment_id']
        
        #(bs,n_sample)
        logit=cross_model(input_id,(input_id!=tokenizer.pad_token_id),segment_id)
        #(bs)
        target=torch.zeros_like(logit,device=logit.device,dtype=torch.long)
        target[:,0]=1
        loss = F.binary_cross_entropy_with_logits(logit,target=target.float())
        step_retrieval_loss+=loss.item()
        loss = loss / args.accum_step
        loss.backward()
    
    grad_norm_cross = torch.nn.utils.clip_grad_norm_([p for p in cross_model.parameters() if p.requires_grad], args.clip)
    if grad_norm_cross >= 1e2:
        logger.info('WARNING : Exploding Poly Gradients {:.2f}'.format(grad_norm_cross))
    cross_optimizer.step()
    cross_scheduler.step()
    cross_optimizer.zero_grad()

    if global_step % args.print_every == 0 and global_step != 0:
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info("Step: %d \t| retrieval_loss: %.3f \t| lr: %.8f \t| %s" % (
            global_step, step_retrieval_loss, cross_scheduler.get_lr()[0], time_str
        ))

def predict_step(global_step,best_metric):
    logits=[]
    labels=[]
    cross_model.eval()
    with torch.no_grad():
        for context_list,candidate_list,label_list in tqdm(eval_loader):
            labels.extend(label_list)
            batch_dict=batcher.retrieve(context_list,candidate_list)
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']
            logit=cross_model(input_id,input_id!=tokenizer.pad_token_id,segment_id,)
            logits.extend(logit.detach().cpu().tolist())

    sessions=[]
    one_session=[]
    score_path=os.path.join(args.output_dir,'{}step_score'.format(global_step))
    f=open(score_path,mode='w',encoding='utf-8')

    for logit,label in zip(logits,labels):
        for i in range(len(logit)):
            f.write(str(logit[i])+'\t'+str(label[i])+'\n')
        if label.count(1)==0:
            continue
        one_session=[(logit[i],label[i]) for i in range(len(logit))]
        sessions.append(np.array(one_session))
    f.close()
    
    from metric import Metrics
    mymetric=Metrics(None,10)
    MAP,MRR,P1,recall1,recall2,recall5=mymetric.evaluate_all_metrics(sessions)

    if not args.debug and not args.predict and recall1>best_metric:
        for file in os.listdir(args.output_dir):
            if file.endswith('model'):
                os.system('rm '+ os.path.join(args.output_dir,file))
        cross_save_path=os.path.join(args.output_dir,'{}step_crossmodel'.format(global_step))
        #torch.save(cross_model.state_dict(),cross_save_path)
    
    logger.info("**********************************")
    logger.info("test results..........")
    logger.info("current best metric {}".format(best_metric))
    logger.info("recall1 recall2 recall5 {},{},{}".format(recall1,recall2,recall5))
    return max(recall1,best_metric)

best_metric = -1.0
print("here is ok 1 ")
if args.predict:
    predict_step(0,best_metric)
    #logger.info("predict result: the f1 between predict knowledge and response: {:.6f}".format(f1))
    exit()
for i in range(args.n_step):
    torch.cuda.empty_cache()
    if not args.naive:
        scheduling_step(i + 1)
    retrieval_step(i + 1)
    if (i + 1) % args.eval_every == 0:
        best_metric = predict_step(i+1,best_metric)

    if i+1 >= 30000 and best_metric < 92.5:
        break
    if i+1 >= 20000 and best_metric < 91.1:
        break