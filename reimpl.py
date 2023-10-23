
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
import random
from tqdm import tqdm
from str2bool import str2bool
import itertools
from datetime import datetime
from transformers import BertTokenizer
from transformers import BertConfig
from transformers import AdamW
from model.retrieval_model import CrossModelv2
from dataset import InferenceDataset,AdversarialDataset
from batcher import EnsembleBatcher

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
parser.add_argument("--debug",default=False,type=str2bool,help='debug mode, using small dataset')
parser.add_argument('--predict',type=str2bool,default=True)
parser.add_argument("--prediction_name",type=str,default='output')

# data
parser.add_argument('--train_file', type=str, help='path to training dataset ')
parser.add_argument('--eval_file', type=str, default='path to evaluation dataset')
parser.add_argument("--rand_cand",type=int,default=0)
parser.add_argument("--inter_cand",type=int,default=0)
parser.add_argument("--keep",help='whether keep the original negative in the training set',type=str2bool,default=True)


# training scheme
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--num_steps', type=int, default=1000000)
parser.add_argument('--accum_step', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--clip', type=float, default=2.0)
parser.add_argument('--schedule', type=str, default='cosine')

parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--warmup_steps', type=int, default=500)
parser.add_argument('--num_epochs', type=int, default=3)

parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--eval_every', type=int, default=2500)

# save
parser.add_argument("--dump_path",type=str,help='the path to dump experiment checkpoints and logs')
parser.add_argument('--exp_name', type=str, default='debug')
parser.add_argument('--log', type=str, default='log')
parser.add_argument('--seed', type=int, default=344)


# input
parser.add_argument("--max_utterance_length",type=int,default=32)
parser.add_argument("--max_context_length",type=int,default=256)
parser.add_argument("--max_cross_length",type=int,default=256)
parser.add_argument("--max_turn",type=str,default=15)

# model path
parser.add_argument('--cross_model_path', type=str,help='the trained checkpoints')
parser.add_argument("--vocab_path",type=str,default='')
parser.add_argument("--config_path",type=str,help='path to the model configuration')

# model architecture
#parser.add_argument("--embedding_dim",default=300)
parser.add_argument("--n_cluster",type=int,default=300)
#parser.add_argument("--n_layer",type=int,default=1)
parser.add_argument("--dropout",type=float,default=0.1)
parser.add_argument("--loss",type=str,default='bce')



args = parser.parse_args()
if args.debug:
    args.print_every=2
    args.eval_every=8
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(0)
out_dir = os.path.join(args.dump_path, args.exp_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
args.out_dir=out_dir
logger.addHandler(logging.FileHandler(os.path.join(args.out_dir, "log"), 'w'))
logger.info("\nParameters:")
for attr, value in sorted(vars(args).items()):
    logger.info("{}={}".format(attr.upper(), value))


# Build dataset
time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("Create dataset begin... | %s " % time_str)

train_dataset=InferenceDataset(args.train_file,debug=args.debug)
train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=InferenceDataset.collate_fn)
train_loader=itertools.cycle(train_loader)
eval_dataset=InferenceDataset(args.eval_file,debug=args.debug)
eval_loader=DataLoader(eval_dataset,batch_size=args.eval_batch_size,shuffle=False,collate_fn=InferenceDataset.collate_fn)

time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.info("Create dataset end... | %s " % time_str)
logger.info('Eval Dataset {}'.format(len(eval_dataset)))
tokenizer=BertTokenizer.from_pretrained(args.vocab_path)
tokenizer.add_special_tokens({'pad_token':'[PAD]','unk_token':'[UNK]','eos_token':'[EOS]'})#,'additional_special_tokens':['INS','DEL','SEAR']})
batcher = EnsembleBatcher(tokenizer,args.max_cross_length,args.max_utterance_length,args.max_context_length,args.max_turn,device,)
configuration=BertConfig.from_json_file(args.config_path)
cross_model=CrossModelv2(configuration)
reloaded=torch.load(args.cross_model_path)
if configuration.vocab_size==30522:
    reloaded['bert.embeddings.word_embeddings.weight']=reloaded['bert.embeddings.word_embeddings.weight'][:30523,:]
elif configuration.vocab_size==21128:
    reloaded['bert.embeddings.word_embeddings.weight']=reloaded['bert.embeddings.word_embeddings.weight'][:21129,:]

cross_model.bert.resize_token_embeddings(len(tokenizer))
cross_model.load_state_dict(reloaded,strict=False)
cross_model.to(device)

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

cross_optimizer = AdamW(cross_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
total_steps = args.num_epochs * (len(train_dataset) / (args.batch_size * args.accum_step))
scheduler = get_linear_schedule_with_warmup(cross_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
if args.schedule == 'linear':
    cross_scheduler = get_linear_schedule_with_warmup(cross_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
elif args.schedule == 'cosine':
    cross_scheduler = get_cosine_schedule_with_warmup(cross_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

def train_step(global_step):
    step_loss={'rs_loss':0.0,'dis_loss':0.0}
    cross_model.train()
    torch.cuda.empty_cache()
    for _ in range(args.accum_step):
        context_list,candidate_list,label_list = next(train_loader)
        bs=len(context_list)
        batch_dict=batcher.crossv1(context_list,candidate_list,mode='train')
        input_id=batch_dict['input_id']
        segment_id=batch_dict['segment_id']
        cross_logit=cross_model(input_id, (input_id!=tokenizer.pad_token_id), segment_id)

        if args.loss=='ce':
            target=torch.zeros(bs,device=device,dtype=torch.long)
            rs_loss=F.cross_entropy(torch.softmax(cross_logit,dim=-1),target)
            #dis_loss=F.cross_entropy(poly_logit,target)*(1-args.alpha) + nn.KLDivLoss()(F.log_softmax(poly_logit/args.temperature,dim=1), F.softmax(cross_logit.detach()/args.temperature,dim=1))* (args.alpha*args.temperature*args.temperature)
        elif args.loss=='bce':
            target=torch.tensor(label_list,dtype=torch.long,device=device)
            rs_loss=F.binary_cross_entropy_with_logits(cross_logit,target.float())
        elif args.loss=='wbce':
            raise NotImplementedError
        
        rs_loss = rs_loss / args.accum_step
        step_loss['rs_loss']+=rs_loss.item()
        rs_loss.backward()

    grad_norm_cross = torch.nn.utils.clip_grad_norm_([p for p in cross_model.parameters() if p.requires_grad], args.clip)
    if grad_norm_cross >= 1e2:
        logger.info('WARNING : Exploding Cross Gradients {:.2f}'.format(grad_norm_cross))
    # if grad_norm_poly >= 1e2:
    #     logger.info('WARNING : Exploding Poly Gradients {:.2f}'.format(grad_norm_poly))
    cross_optimizer.step()
    cross_scheduler.step()
    cross_optimizer.zero_grad()

    if global_step % args.print_every == 0 and global_step != 0:
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info("Step: %d \t| rs_loss: %.3f \t|feature_loss: %.3f \t| lr: %.8f \t| %s" % (
            global_step, step_loss['rs_loss'], step_loss['dis_loss'],cross_scheduler.get_lr()[0], time_str
        ))

def predict_step(global_step,best_metric):
    #hypothesis=[]
    count=0
    logits=[]
    labels=[]
    sessions=[]
    cross_model.eval()

    with torch.no_grad():
        for context_list,candidate_list,labellist in tqdm(eval_loader):
            bs=len(context_list)
            count+=bs
            batch_dict=batcher.crossv1(context_list,candidate_list,mode='train')
            input_id=batch_dict['input_id']
            segment_id=batch_dict['segment_id']
            logit=cross_model(input_id,(input_id!=tokenizer.pad_token_id),segment_id)
            logits.extend(logit.detach().cpu().tolist())
            labels.extend(labellist)
            if count%10000==0:
                logger.info('eval finish {}'.format(count))
   
    # for logit,label in zip(logits,labels):
    #     if label.count(1)!=0:
    #         one_session=[(logit[i],label[i]) for i in range(len(logit))]
    #         sessions.append(np.array(one_session))
    if args.predict:
        f=open(os.path.join(args.out_dir,args.prediction_name),'w')
    else:
        f=open(os.path.join(args.out_dir,'{}step_score_file'.format(global_step)),'w')
    for logit,label in zip(logits,labels):
        assert len(logit)==len(label)
        for i in range(len(logit)):
            f.write(str(logit[i])+'\t'+str(label[i])+'\n')
    f.close()

    one_session=[]
    golden=0
    for i in range(len(logits)):
        golden=labels[i].count(1)
        if golden>=1:
            sessions.append(np.array([(lo,la) for lo,la in zip(logits[i],labels[i])]))

    from metric import Metrics
    mymetric=Metrics(None)
    MAP,MRR,P1,recall1,recall2,recall5=mymetric.evaluate_all_metrics(sessions)
    logger.info("**********************************")
    logger.info("test results..........")
    logger.info("MAP {}".format(MAP))
    logger.info("MRR {}".format(MRR))
    logger.info("P1 {}".format(P1))
    logger.info("recall1 {}".format(recall1))
    logger.info("recall1 {}".format(recall2))
    logger.info("recall1 {}".format(recall5))

    if not args.debug and not args.predict and recall1 > best_metric:
        for file in os.listdir(args.out_dir):
            if file.endswith('crossmodel'):
                os.system('rm  ' + os.path.join(args.out_dir,file))
        cross_save_path=os.path.join(args.out_dir,'{}step_crossmodel'.format(global_step))
        torch.save(cross_model.state_dict(),cross_save_path)
    #logger.info("prediction precision={}".format(precision))
    return recall1

best_metric = -1
patience=0
if args.predict:
    predict_step(0,best_metric)
    #logger.info("predict result: the f1 between predict knowledge and response: {:.6f}".format(f1))
    exit()
for i in range(args.num_steps):
    train_step(i + 1)
    if (i + 1) % args.eval_every == 0 :
        metric=predict_step(i+1,best_metric)
        if metric > best_metric:
            best_metric=metric
            patience=0
        else:
            patience+=1
