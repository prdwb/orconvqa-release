from __future__ import absolute_import, division, print_function

import json
import logging
import math
import collections
import linecache
import numpy as np
from io import open
from tqdm import tqdm
from torch.utils.data import Dataset

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
# from utils_squad_evaluate import find_all_best_thresh_v2, make_qid_to_has_ans, get_raw_scores

logger = logging.getLogger(__name__)

class RetrieverInputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class RetrieverInputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    
    
class RetrieverDataset(Dataset):
    def __init__(self, filename, tokenizer, 
                 load_small, history_num, prepend_history_questions=True, 
                 prepend_history_answers=False,
                 query_max_seq_length=128, passage_max_seq_length=384,
                 is_pretraining=False, given_query=False, 
                 given_passage=False, only_positive_passage=True, 
                 include_first_for_retriever=False):
        
        self._filename = filename
        self._tokenizer = tokenizer
        self._load_small = load_small
        self._history_num = history_num
        self._query_max_seq_length = query_max_seq_length
        self._passage_max_seq_length = passage_max_seq_length
        self._prepend_history_questions = prepend_history_questions
        self._prepend_history_answers = prepend_history_answers
        
        # if given query:
            # if pretraining: using rewrite as question
            # else: using concat of question
        self._is_pretraining = is_pretraining 
        self._given_query = given_query
        self._given_passage = given_passage
        
        # if we only pass the positive passages to the model
        # the rest of the passges in the batch are considered as negatives
        self._only_positive_passage = only_positive_passage
        
        self._include_first_for_retriever = include_first_for_retriever
        
        self._total_data = 0      
        if self._load_small:
            self._total_data = 50
        else:
            with open(filename, "r") as f:
                self._total_data = len(f.readlines())
                
    def __len__(self):
        return self._total_data
                
    def __getitem__(self, idx):
        """read a line of preprocessed open-retrieval quac file into a quac example"""
        line = linecache.getline(self._filename, idx + 1)    
        entry = json.loads(line.strip())
        qas_id = entry["qid"]
        retrieval_labels = entry['retrieval_labels']
        
        return_feature_dict = {}
        
        if self._given_query:
            if self._is_pretraining:
                question_text_for_retriever = entry["rewrite"]
            else:
                orig_question_text = entry["question"]
                history = entry['history']
                question_text_list = []
                if self._history_num > 0:
                    for turn in history[- self._history_num :]:
                        if self._prepend_history_questions:
                            question_text_list.append(turn['question'])
                        if self._prepend_history_answers:
                            question_text_list.append(turn['answer']['text'])
                question_text_list.append(orig_question_text)
                question_text = ' [SEP] '.join(question_text_list)
                question_text_for_retriever = question_text
    
                # include the first question in addition to history_num for retriever (not reader)
                if self._include_first_for_retriever and len(history) > 0:
                    first_question = history[0]['question']
                    if first_question != question_text_list[0]:                    
                        question_text_for_retriever = first_question + ' [SEP] ' + question_text
                        
                # print('question_text_for_retriever', question_text_for_retriever)
                # print('question_text', question_text)
                        
            # print('question_text_for_retriever', question_text_for_retriever)
            query_example = RetrieverInputExample(guid=qas_id, text_a=question_text_for_retriever)
            query_feature = retriever_convert_example_to_feature(query_example, self._tokenizer, 
                                                                 max_length=self._query_max_seq_length)
            query_feature_dict = {'query_input_ids': np.asarray(query_feature.input_ids), 
                                  'query_token_type_ids': np.asarray(query_feature.token_type_ids), 
                                  'query_attention_mask': np.asarray(query_feature.attention_mask), 
                                  'qid': qas_id}
            # during fine-tuning, we also return the query text for training reader
            if not self._is_pretraining:
                query_feature_dict['question_text'] = question_text
                query_feature_dict['answer_text'] = entry['answer']['text']
                query_feature_dict['answer_start'] = entry['answer']['answer_start']
            
            return_feature_dict.update(query_feature_dict)
        
        if self._given_passage:
            passages = entry['evidences']
            
            if self._only_positive_passage:
                postive_idx = np.argmax(retrieval_labels)
                passage = passages[postive_idx]

                example_id = '{}_{}'.format(qas_id, postive_idx)
                passage_example = RetrieverInputExample(
                                        guid=example_id,
                                        text_a=passage,
                                        label=1)

                passage_feature = retriever_convert_example_to_feature(passage_example, self._tokenizer, 
                                                                       max_length=self._passage_max_seq_length)
                passage_feature_dict = {'passage_input_ids': np.asarray(passage_feature.input_ids), 
                                 'passage_token_type_ids': np.asarray(passage_feature.token_type_ids), 
                                 'passage_attention_mask': np.asarray(passage_feature.attention_mask),
                                 'retrieval_label': passage_feature.label, 
                                 'example_id': example_id}
                return_feature_dict.update(passage_feature_dict)
                
            else:
                batch = []
                passage_examples = []
                for i, (passage, retrieval_label) in enumerate(zip(passages, retrieval_labels)):
                    example_id = '{}_{}'.format(qas_id, i)
                    passage_example = RetrieverInputExample(
                                            guid=example_id,
                                            text_a=passage,
                                            label=retrieval_label)

                    passage_feature = retriever_convert_example_to_feature(passage_example, self._tokenizer, 
                                                                           max_length=self._passage_max_seq_length)
                    batch_feature = {'passage_input_ids': np.asarray(passage_feature.input_ids), 
                                     'passage_token_type_ids': np.asarray(passage_feature.token_type_ids), 
                                     'passage_attention_mask': np.asarray(passage_feature.attention_mask),
                                     'retrieval_label': passage_feature.label, 
                                     'example_id': example_id}

                    batch.append(batch_feature)

                collated = {}
                keys = batch[0].keys()
                for key in keys:
                    if key != 'example_id':
                        collated[key] = np.vstack([dic[key] for dic in batch])
                if 'example_id' in keys:
                    collated['example_id'] = [dic['example_id'] for dic in batch]

                return_feature_dict.update(collated)

        return return_feature_dict
    
class GenPassageRepDataset(Dataset):
    def __init__(self, filename, tokenizer, 
                 load_small, passage_max_seq_length=386):
        
        self._filename = filename
        self._tokenizer = tokenizer
        self._load_small = load_small  
        self._passage_max_seq_length = passage_max_seq_length
                
        self._total_data = 0      
        if self._load_small:
            self._total_data = 100
        else:
            with open(filename, "r") as f:
                self._total_data = len(f.readlines())
                
    def __len__(self):
        return self._total_data
                
    def __getitem__(self, idx):
        """read a line of preprocessed open-retrieval quac file into a quac example"""
        line = linecache.getline(self._filename, idx + 1)    
        entry = json.loads(line.strip())
        example_id = entry["id"]
        passage = entry['text']
        
        passage_example = RetrieverInputExample(guid=example_id, text_a=passage)
        passage_feature = retriever_convert_example_to_feature(passage_example, self._tokenizer, 
                                                               max_length=self._passage_max_seq_length)
        batch_feature = {'passage_input_ids': np.asarray(passage_feature.input_ids), 
                         'passage_token_type_ids': np.asarray(passage_feature.token_type_ids), 
                         'passage_attention_mask': np.asarray(passage_feature.attention_mask),
                         'example_id': example_id}

        return batch_feature

def retriever_convert_example_to_feature(example, tokenizer,
                                      max_length=512,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """


    inputs = tokenizer.encode_plus(
        example.text_a,
        example.text_b,
        add_special_tokens=True,
        max_length=max_length,
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

    if False:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        logger.info("label: %s" % (example.label))

    feature = RetrieverInputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=example.label)

    return feature