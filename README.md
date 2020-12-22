# Open-Retrieval Conversational Question Answering

This repo contains the code and data for our paper [Open-Retrieval Conversational Question Answering](https://arxiv.org/pdf/2005.11364.pdf).


### Data and checkpoints
Download [here](https://ciir.cs.umass.edu/downloads/ORConvQA/). The data is distributed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license. Our data enhances QuAC by adapting it to an open-retrieval setting. It is an aggregation of three existing datasets: (1) the [QuAC](http://quac.ai/) dataset that offers information-seeking conversations, (2) the [CANARD](https://sites.google.com/view/qanta/projects/canard) dataset that consists of context-independent rewrites of QuAC questions, and (3) the Wikipedia corpus that serves as the knowledge source of answering questions.

OR-QuAC files:
* OR-QuAC dev/test files in QuAC format. This is used in evaluation with the QuAC evaluation script.
* Preprocessed OR-QuAC train/dev/test files. Model input. Each line is an example. The `evidence` field is not used for concurrent learning. Only the positive passage in `evidence` is used for retriever pretraining. Other passages are retrieved by TF-IDF. In dev/test, there might be no gold passages in this field.
* qrels.txt. Passage relevance derived from QuAC. Note this is partial relevance since passages beyond the gold ones may also contain answers.
* all_blocks.txt. This file has all passages in the collection.

If you would like to use our pretrained retriever, also download:
* passage_reps.pkl. This contains all passage representations.
* passage_ids.pkl. This contains passage ids for passage reps.
* pretrained retriever. The retriever checkpoint used in concurrent learning.

Optional
* dev_blocks.txt. This is a small collection for selecting the best retriever model if you want to train your own retriever. You can also build your own small collection.
* pipeline checkpoint. This is not necessary to download if you want to train your own model.


### Concurrent learning of the full model
We use two GPUs for this part and one of the GPUs is for MIPS.

```
python3 train_pipeline.py 
    --train_file=path_to_preprocessed_train_file
    --dev_file=path_to_preprocessed_dev_file
    --test_file=path_to_preprocessed_test_file
    --orig_dev_file=path_to_dev_file_in_quac_format
    --orig_test_file=path_to_test_file_in_quac_format
    --qrels=path_to_qrels_txt 
    --blocks_path=path_to_all_blocks_txt 
    --passage_reps_path=path_to_passage_reps_pkl 
    --passage_ids_path=path_to_passage_ids_pkl 
    --output_dir=output_dir
    --load_small=False (set to True to load a small amount of data only for testing purposes)
    --history_num=6 (how many history turns to prepend)
    --do_train=True 
    --do_eval=True 
    --do_test=True 
    --per_gpu_train_batch_size=2 
    --per_gpu_eval_batch_size=4 
    --learning_rate=5e-5 
    --num_train_epochs=3.0 
    --logging_steps=5 
    --save_steps=5000 
    --overwrite_output_dir=False 
    --eval_all_checkpoints=True 
    --fp16=True 
    --retriever_cache_dir=path_to_huggingface_albert_v1_cache (optional)
    --retrieve_checkpoint=path_to_retriever_checkpoint/checkpoint-xxx 
    --retrieve_tokenizer_dir=path_to_retriever_checkpoint
    --top_k_for_retriever=100 (use how many retrieved passages to update the question encoder in the retriever)
    --use_retriever_prob=True (use retriever score in overall score)
    --reader_cache_dir=path_to_huggingface_bert_cache (optional)
    --qa_loss_factor=1.0 
    --retrieval_loss_factor=1.0 
    --top_k_for_reader=5 (retrieve how many passages for reader)
    --include_first_for_retriever=True 
    --use_rerank_prob=True (use reranker score in overall score)
    --early_loss=True (fine tune the question encoder in the retriever)
    --max_answer_length=40
```


### Retriever pretraining
Skip this step if you would like to use our pretrained retriever. We use distributed training but it is optional.  

In parallel with our work, Facebook released a [dense passage retriever](https://github.com/facebookresearch/DPR) that has a similar architecture with our retriever but has a better training approach. They include a single BM25-retrieved negative passage for every question in addition to the in-batch negatives. They show this brings significant gains.

```
python3 -m torch.distributed.launch --nproc_per_node 4 train_retriever.py 
    --train_file=path_to_preprocessed_train_file
    --dev_file=path_to_preprocessed_dev_file
    --test_file=path_to_preprocessed_test_file
    --model_name_or_path=albert-base-v1 
    --output_dir=output_dir
    --tokenizer_name=albert-base-v1 
    --cache_dir=path_to_huggingface_albert_v1_cache (optional)
    --per_gpu_train_batch_size=16 
    --per_gpu_eval_batch_size=16 
    --learning_rate=5e-5 
    --num_train_epochs=12.0 
    --logging_steps=5 
    --save_steps=5000 
    --load_small=False 
    --fp16=True 
    --overwrite_output_dir=False 
    --do_train=True 
    --do_eval=True 
    --do_test=False 
    --eval_all_checkpoints=True 
    --given_query=True 
    --given_passage=True 
    --is_pretraining=True 
    --gradient_accumulation_steps=1 
    --gen_passage_rep=False (False: pretraining, True: generate all passage reps with a checkpoint)
    --retrieve=False 
    --only_positive_passage=True (always set to True, use in-batch negatives)
    --qrels=path_to_qrels_txt
    --gen_passage_rep_input=path_to_dev_blocks_txt 
    --query_max_seq_length=128 
    --passage_max_seq_length=384 
    --gen_passage_rep_output=output_dir/dev_blocks.txt (a temporary file)
```

### Generate passage representations with pretrained retriever
Skip this step if you are using our pretrained retriever since the passage reps are available to download.

```
python3 train_retriever.py 
    --model_name_or_path=albert-base-v1 
    --output_dir=path_to_retriever_checkpoint
    --tokenizer_name=albert-base-v1 
    --cache_dir=path_to_huggingface_albert_v1_cache (optional)
    --per_gpu_eval_batch_size=300 
    --load_small=False 
    --fp16=True 
    --overwrite_output_dir=True 
    --gen_passage_rep=True 
    --retrieve=False 
    --only_positive_passage=True 
    --gen_passage_rep_input=path_to_all_blocks_txt (in practice, use a split of all_blocks.txt for parallel processing)
    --gen_passage_rep_output=path_to_store_the_passage_reps
    --passage_max_seq_length=384 
    --retrieve_checkpoint=path_to_retriever_checkpoint/checkpoint-xxx/
```


### Environment
* Install [Huggingface Transformers](https://github.com/huggingface/transformers), [Faiss](https://github.com/facebookresearch/faiss), and [pytrec_eval](https://github.com/cvangysel/pytrec_eval)
* Developed with Python 3.7, Torch 1.2.0, and Transformers 2.3.0


### Citation
```
@inproceedings{orconvqa,
  title={{Open-Retrieval Conversational Question Answering}},
  author={Chen Qu and Liu Yang and Cen Chen and Minghui Qiu and W. Bruce Croft and Mohit Iyyer},
  booktitle={SIGIR},
  year={2020}
}
```
