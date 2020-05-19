#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse
import logging
import os
import random
import glob
import timeit
import json
import faiss

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import pytrec_eval

from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer, AlbertConfig, AlbertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from retriever_utils import RetrieverDataset, GenPassageRepDataset
from modeling import BertForRetrieverOnlyPositivePassage, AlbertForRetrieverOnlyPositivePassage


# In[2]:


logger = logging.getLogger(__name__)

ALL_MODELS = list(BertConfig.pretrained_config_archive_map.keys())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForRetrieverOnlyPositivePassage, BertTokenizer),
    'albert': (AlbertConfig, AlbertForRetrieverOnlyPositivePassage, AlbertTokenizer),
}


# In[3]:


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# In[4]:


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    args.warmup_steps = int(t_total * args.warmup_portion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch", disable=args.local_rank not in [-1, 0])
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = {k: v.to(args.device) for k, v in batch.items() if k not in ['example_id', 'qid']}
            inputs = {}
            if args.given_query:
                inputs['query_input_ids'] = batch['query_input_ids']
                inputs['query_attention_mask'] = batch['query_attention_mask']        
                inputs['query_token_type_ids'] = batch['query_token_type_ids']
            if args.given_passage:
                inputs['passage_input_ids'] = batch['passage_input_ids']
                inputs['passage_attention_mask'] = batch['passage_attention_mask']        
                inputs['passage_token_type_ids'] = batch['passage_token_type_ids']
                inputs['retrieval_label'] = batch['retrieval_label']
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            if args.given_query and args.given_passage:
                loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar(
                        'lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(
                        output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


# In[5]:


def evaluate(args, model, tokenizer, prefix=""):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    predict_dir = os.path.join(args.output_dir, 'predictions')
    if not os.path.exists(predict_dir) and args.local_rank in [-1, 0]:
        os.makedirs(predict_dir)

    passage_ids, passage_reps = gen_passage_rep(args, model, tokenizer)
    passage_reps = np.asarray(passage_reps, dtype='float32')
    qids, query_reps = retrieve(args, model, tokenizer)
    query_reps = np.asarray(query_reps, dtype='float32')
        
    index = faiss.IndexFlatIP(args.proj_size)
    index.add(passage_reps)
    D, I = index.search(query_reps, 5)
    
    # print(qids, query_reps, passage_ids, passage_reps, D, I)

    run = {}
    for qid, retrieved_ids, scores in zip(qids, I, D):
        run[qid] = {passage_ids[retrieved_id]: float(score) for retrieved_id, score in zip(retrieved_ids, scores)}

    with open(args.qrels) as handle:
        qrels = json.load(handle)
    evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {'recip_rank', 'recall'})
    metrics = evaluator.evaluate(run)
    mrr_list = [v['recip_rank'] for v in metrics.values()]
    recall_list = [v['recall_5'] for v in metrics.values()]
    eval_metrics = {'MRR': np.average(mrr_list), 'Recall': np.average(recall_list)}

    return eval_metrics


# In[6]:


def gen_passage_rep(args, model, tokenizer):
    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    DatasetClass = GenPassageRepDataset
    dataset = DatasetClass(args.gen_passage_rep_input, tokenizer,
                           args.load_small, 
                           passage_max_seq_length=args.passage_max_seq_length)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(
    #     dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')

    # Eval!
    logger.info("***** Gem passage rep *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    run_dict = {}
    start_time = timeit.default_timer()
    fout = open(args.gen_passage_rep_output, 'w')
    passage_ids = []
    passage_reps_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        example_ids = np.asarray(
            batch['example_id']).reshape(-1).tolist()
        passage_ids.extend(example_ids)
        batch = {k: v.to(args.device)
                 for k, v in batch.items() if k != 'example_id'}
        with torch.no_grad():
            inputs = {}
            inputs['passage_input_ids'] = batch['passage_input_ids']
            inputs['passage_attention_mask'] = batch['passage_attention_mask']
            inputs['passage_token_type_ids'] = batch['passage_token_type_ids']
            outputs = model(**inputs)
            passage_reps = outputs[0]
            passage_reps_list.extend(to_list(passage_reps))
        
        # with open(args.gen_passage_rep_output, 'w') as fout:
        for example_id, passage_rep in zip(example_ids, to_list(passage_reps)):
            fout.write(json.dumps({'id': example_id, 'rep': passage_rep}) + '\n')
    fout.close()
    return passage_ids, passage_reps_list


# In[7]:


def retrieve(args, model, tokenizer, prefix=''):
    if prefix == 'test':
        eval_file = args.test_file
    else:
        eval_file = args.dev_file

    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    DatasetClass = RetrieverDataset
    dataset = DatasetClass(eval_file, tokenizer,
                           args.load_small, args.history_num,
                           query_max_seq_length=args.query_max_seq_length,
                           passage_max_seq_length=args.passage_max_seq_length,
                           is_pretraining=args.is_pretraining,
                           given_query=True,
                           given_passage=False)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(
    #     dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')
        
    # Eval!
    logger.info("***** Retrieve {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_qids = []
    all_query_reps = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        qids = np.asarray(
            batch['qid']).reshape(-1).tolist()
        batch = {k: v.to(args.device)
                 for k, v in batch.items() if k not in ['example_id', 'qid']}
        with torch.no_grad():
            inputs = {}
            inputs['query_input_ids'] = batch['query_input_ids']
            # print(inputs['query_input_ids'], inputs['query_input_ids'].size())
            inputs['query_attention_mask'] = batch['query_attention_mask']
            inputs['query_token_type_ids'] = batch['query_token_type_ids']
            outputs = model(**inputs)
            query_reps = outputs[0]

        all_qids.extend(qids)
        all_query_reps.extend(to_list(query_reps))

    return all_qids, all_query_reps


# In[8]:


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--train_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/preprocessed/train.txt',
                    type=str, required=False,
                    help="open retrieval quac json for training. ")
# parser.add_argument("--train_file", default='/mnt/scratch/chenqu/orconvqa/v5/test_retriever/google_nq+quac.txt',
#                     type=str, required=False,
#                     help="open retrieval quac json for training. ")
parser.add_argument("--dev_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/preprocessed/dev.txt',
                    type=str, required=False,
                    help="open retrieval quac json for predictions.")
parser.add_argument("--test_file", default='/mnt/scratch/chenqu/orconvqa/v5/quac_canard/preprocessed/test.txt',
                    type=str, required=False,
                    help="open retrieval quac json for predictions.")
parser.add_argument("--model_type", default='albert', type=str, required=False,
                    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
# parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, required=False,
#                     help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
parser.add_argument("--model_name_or_path", default='albert-base-v1', type=str, required=False,
                    help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
parser.add_argument("--output_dir", default='/mnt/scratch/chenqu/orconvqa_output/retriever_release_test', type=str, required=False,
                    help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument("--qrels", default='/mnt/scratch/chenqu/orconvqa/v5/retrieval/qrels.txt', type=str, required=False,
                    help="qrels to evaluate open retrieval")

parser.add_argument("--given_query", default=True, type=str2bool,
                    help="Whether query is given.")
parser.add_argument("--given_passage", default=True, type=str2bool,
                    help="Whether passage is given.")
parser.add_argument("--is_pretraining", default=True, type=str2bool,
                    help="Whether is pretraining.")
parser.add_argument("--only_positive_passage", default=True, type=str2bool,
                    help="we only pass the positive passages, the rest of the passges in the batch are considered as negatives")
parser.add_argument("--gen_passage_rep", default=False, type=str2bool,
                    help="generate passage representations for all ")
parser.add_argument("--retrieve_checkpoint", 
                    default='/mnt/scratch/chenqu/orconvqa_output/retriever_release_test/checkpoint-5917', type=str,
                    help="generate query/passage representations with this checkpoint")
parser.add_argument("--gen_passage_rep_input", 
                    default='/mnt/scratch/chenqu/orconvqa/v5/test_retriever/dev_blocks.txt', type=str,
                    help="generate passage representations for this file that contains passages")
parser.add_argument("--gen_passage_rep_output", 
                    default='/mnt/scratch/chenqu/orconvqa_output/retriever_release_test/dev_blocks.txt', type=str,
                    help="passage representations")
parser.add_argument("--retrieve", default=False, type=str2bool,
                    help="generate query reps and retrieve passages")

# Other parameters
parser.add_argument("--config_name", default="", type=str,
                    help="Pretrained config name or path if not the same as model_name")
# parser.add_argument("--tokenizer_name", default="bert-base-uncased", type=str,
#                     help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--tokenizer_name", default="albert-base-v1", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="/mnt/scratch/chenqu/huggingface_cache/albert_v1/", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")

parser.add_argument("--query_max_seq_length", default=128, type=int,
                    help="The maximum input sequence length of query (125 + [CLS] + [SEP])."
                         "125 is the max question length in the reader.")
parser.add_argument("--passage_max_seq_length", default=384, type=int,
                    help="The maximum input sequence length of passage (384 + [CLS] + [SEP]).")
parser.add_argument("--proj_size", default=128, type=int,
                    help="The size of the query/passage rep after projection of [CLS] rep.")
parser.add_argument("--do_train", default=True, type=str2bool,
                    help="Whether to run training.")
parser.add_argument("--do_eval", default=True, type=str2bool,
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_test", default=False, type=str2bool,
                    help="Whether to run eval on the test set.")
parser.add_argument("--evaluate_during_training", default=False, type=str2bool,
                    help="Rul evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", default=True, type=str2bool,
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=12, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=300, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=2.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--warmup_portion", default=0.1, type=float,
                    help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps ")
parser.add_argument("--verbose_logging", action='store_true',
                    help="If true, all of the warnings related to data processing will be printed. "
                         "A number of warnings are expected for a normal SQuAD evaluation.")

parser.add_argument('--logging_steps', type=int, default=1,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=20,
                    help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints", default=True, type=str2bool,
                    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", default=False, type=str2bool,
                    help="Whether not to use CUDA when available")
parser.add_argument('--overwrite_output_dir', default=True, type=str2bool,
                    help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--fp16', default=False, type=str2bool,
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--server_ip', type=str, default='',
                    help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='',
                    help="Can be used for distant debugging.")

parser.add_argument("--load_small", default=True, type=str2bool, required=False,
                    help="whether to load just a small portion of data during development")
parser.add_argument("--history_num", default=1, type=int, required=False,
                    help="number of history turns to use")
parser.add_argument("--num_workers", default=4, type=int, required=False,
                    help="number of workers for dataloader")

args, unknown = parser.parse_known_args()

if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError(
        "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

# Setup distant debugging if needed
if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(
        address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    torch.cuda.set_device(0)
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = 1
args.device = device

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
               args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

# Set seed
set_seed(args)

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

args.model_type = args.model_type.lower()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                      cache_dir=args.cache_dir if args.cache_dir else None)
config.proj_size = args.proj_size

tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                            do_lower_case=args.do_lower_case,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
model = model_class.from_pretrained(args.model_name_or_path,
                                    from_tf=bool(
                                        '.ckpt' in args.model_name_or_path),
                                    config=config,
                                    cache_dir=args.cache_dir if args.cache_dir else None)
# model = model_class.from_pretrained(args.retrieve_checkpoint,
#                                     from_tf=bool(
#                                         '.ckpt' in args.model_name_or_path),
#                                     config=config,
#                                     cache_dir=args.cache_dir if args.cache_dir else None)

if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

model.to(args.device)

logger.info("Training/evaluation parameters %s", args)

# Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
# Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
# remove the need for this code, but it is still valid.
if args.fp16:
    try:
        import apex
        apex.amp.register_half_function(torch, 'einsum')
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

if args.retrieve:
    args.gen_passage_rep = False
    args.do_train = False
    args.do_eval = False
    args.do_test = False
    args.given_query = True
    args.given_passage = False

if args.gen_passage_rep:
    args.do_train = False
    args.do_eval = False
    args.do_test = False

# if args.do_eval or args.evaluate_during_training or args.retrieve:
#     eval_qrels = {}
#     with open(args.dev_file) as fin:
#         for line in fin:
#             instance = json.loads(line.strip())
#             qid = instance['qid']
#             evidences, retrieval_labels = instance['evidences'], instance['retrieval_labels']
#             eval_qrels[qid] = {}
#             for i, (evidence, retrieval_label) in enumerate(zip(evidences, retrieval_labels)):
#                 doc_id = '{}_{}'.format(qid, i)
#                 eval_qrels[qid][doc_id] = retrieval_label

#     evaluator = pytrec_eval.RelevanceEvaluator(
#         eval_qrels, {'recip_rank', 'recall'})

# if args.do_test or args.retrieve:
#     test_qrels = {}
#     with open(args.test_file) as fin:
#         for line in fin:
#             instance = json.loads(line.strip())
#             qid = instance['qid']
#             evidences, retrieval_labels = instance['evidences'], instance['retrieval_labels']
#             test_qrels[qid] = {}
#             for i, (evidence, retrieval_label) in enumerate(zip(evidences, retrieval_labels)):
#                 doc_id = '{}_{}'.format(qid, i)
#                 test_qrels[qid][doc_id] = retrieval_label

#     test_evaluator = pytrec_eval.RelevanceEvaluator(
#         test_qrels, {'recip_rank', 'recall'})

# Training
if args.do_train:
    DatasetClass = RetrieverDataset
    train_dataset = DatasetClass(args.train_file, tokenizer,
                                 args.load_small, args.history_num,
                                 query_max_seq_length=args.query_max_seq_length,
                                 passage_max_seq_length=args.passage_max_seq_length,
                                 is_pretraining=True,
                                 given_query=True,
                                 given_passage=True, 
                                 only_positive_passage=args.only_positive_passage)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s",
                global_step, tr_loss)

# Save the trained model and the tokenizer
if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, 'module') else model
    final_checkpoint_output_dir = os.path.join(
        args.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(final_checkpoint_output_dir):
        os.makedirs(final_checkpoint_output_dir)

    model_to_save.save_pretrained(final_checkpoint_output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(
        final_checkpoint_output_dir, 'training_args.bin'))

    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(
        final_checkpoint_output_dir, force_download=True)
    tokenizer = tokenizer_class.from_pretrained(
        args.output_dir, do_lower_case=args.do_lower_case)
    model.to(args.device)


# In[9]:


# Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory

results = {}
max_mrr = 0.0
best_metrics = {}
if args.do_eval and args.local_rank in [-1, 0]:
    tokenizer = tokenizer_class.from_pretrained(
        args.output_dir, do_lower_case=args.do_lower_case)
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
        checkpoints = list(os.path.dirname(c) for c in sorted(
            glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
#         logging.getLogger("transformers.modeling_utils").setLevel(
#             logging.WARN)  # Reduce model loading logs

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
        # Reload the model
        global_step = checkpoint.split(
            '-')[-1] if len(checkpoint) > 1 else ""
        print(global_step, 'global_step')
        model = model_class.from_pretrained(
            checkpoint, force_download=True)
        model.to(args.device)

        # Evaluate
        result = evaluate(args, model, tokenizer, prefix=global_step)
        if result['MRR'] > max_mrr:
            max_mrr = result['MRR']
            best_metrics['MRR'] = result['MRR']
            best_metrics['Recall'] = result['Recall']
            best_metrics['global_step'] = global_step

        for key, value in result.items():
            tb_writer.add_scalar(
                'eval_{}'.format(key), value, global_step)

        result = dict((k + ('_{}'.format(global_step) if global_step else ''), v)
                      for k, v in result.items())
        results.update(result)

    best_metrics_file = os.path.join(
        args.output_dir, 'predictions', 'best_metrics.json')
    with open(best_metrics_file, 'w') as fout:
        json.dump(best_metrics, fout)
        
    all_results_file = os.path.join(
        args.output_dir, 'predictions', 'all_results.json')
    with open(all_results_file, 'w') as fout:
        json.dump(results, fout)

    logger.info("Results: {}".format(results))
    logger.info("best metrics: {}".format(best_metrics))


# In[10]:


if args.do_test and args.local_rank in [-1, 0]:
    best_global_step = best_metrics['global_step']
    best_checkpoint = os.path.join(
        args.output_dir, 'checkpoint-{}'.format(best_global_step))
    logger.info("Test the best checkpoint: %s", best_checkpoint)

    model = model_class.from_pretrained(
        best_checkpoint, force_download=True)
    model.to(args.device)

    # Evaluate
    result = evaluate(args, model, tokenizer, prefix='test')

    test_metrics_file=os.path.join(
        args.output_dir, 'predictions', 'test_metrics.json')
    with open(test_metrics_file, 'w') as fout:
        json.dump(result, fout)

    logger.info("Test Result: {}".format(result))


# In[11]:


if args.gen_passage_rep and args.local_rank in [-1, 0]:
    tokenizer = tokenizer_class.from_pretrained(
        args.output_dir, do_lower_case=args.do_lower_case)
    logger.info("Gen passage rep with: %s", args.retrieve_checkpoint)

    model = model_class.from_pretrained(
        args.retrieve_checkpoint, force_download=True)
    model.to(args.device)

    # Evaluate
    gen_passage_rep(args, model, tokenizer)
    
    logger.info("Gen passage rep complete")


# In[12]:


if args.retrieve and args.local_rank in [-1, 0]:
    tokenizer = tokenizer_class.from_pretrained(
        args.output_dir, do_lower_case=args.do_lower_case)
    logger.info("Retrieve with: %s", args.retrieve_checkpoint)
    model = model_class.from_pretrained(
        args.retrieve_checkpoint, force_download=True)
    model.to(args.device)

    # Evaluate
    qids, query_reps = retrieve(args, model, tokenizer)
    query_reps = np.asarray(query_reps, dtype='float32')
       
    logger.info("Gen query rep complete")


# In[13]:


# passage_ids, passage_reps = [], []
# with open(args.gen_passage_rep_output) as fin:
#     for line in tqdm(fin):
#         dic = json.loads(line.strip())
#         passage_ids.append(dic['id'])
#         passage_reps.append(dic['rep'])
# passage_reps = np.asarray(passage_reps, dtype='float32')

