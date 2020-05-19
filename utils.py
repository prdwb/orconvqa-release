# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Load QuAC dataset. """

from __future__ import absolute_import, division, print_function

import json
import logging
import math
import collections
import linecache
import numpy as np
from io import open
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
# from utils_squad_evaluate import find_all_best_thresh_v2, make_qid_to_has_ans, get_raw_scores

logger = logging.getLogger(__name__)


class QuacExample(object):
    """
    A single training/test example for the QuAC dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 example_id,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None, 
                 followup=None, 
                 yesno=None, 
                 retrieval_label=None, 
                 history=None):
        self.example_id = example_id
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.followup = followup
        self.yesno = yesno
        self.retrieval_label = retrieval_label
        self.history = history

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "example_id: %s" % (self.example_id)
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        s += ', followup: {}'.format(self.followup)
        s += ', yesno: {}'.format(self.yesno)
        if self.retrieval_label:
            s += ', retrieval_label: {}'.format(self.retrieval_label)
        s += ', history: {}'.format(self.history)
            
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_id,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None, 
                 retrieval_label=None):
        # we have exactly 1 feature for every example,
        # so the unique id is the same with the example id
        self.unique_id = unique_id 
        self.example_id = example_id
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.retrieval_label = retrieval_label

class LazyQuacDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer, 
                 load_small, history_num, prepend_history_questions, 
                 prepend_history_answers, embed_history_answers, 
                 is_training=True):
        
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self._load_small = load_small
        self._history_num = history_num
        self._is_training = is_training
        self._prepend_history_questions = prepend_history_questions
        self._prepend_history_answers = prepend_history_answers
        self._embed_history_answers = embed_history_answers
        
        self.all_examples = {}
        self.all_features = {}
        
        self._total_data = 0      
        if self._load_small:
            self._total_data = 100
        else:
            with open(filename, "r") as f:
                self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        """read a line of preprocessed open-retrieval quac file into a quac example"""
        
        line = linecache.getline(self._filename, idx + 1)        
        entry = json.loads(line.strip())
        example_id = entry['unique_id']
        
        paragraph_text = entry['evidence']
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
            
        qas_id = entry["qid"]
        
        
        orig_question_text = entry["question"]
        
        # TODO: fix the bug when history_num = 0
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
        
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False        
        
        if self._is_training:

            if entry['answer']['text'] in ['CANNOTANSWER', 'NOTRECOVERED']:
                is_impossible = True

            if not is_impossible:
                answer = entry['answer']
                orig_answer_text = answer["text"]
                answer_offset = answer["answer_start"]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset + answer_length - 1]
                # Only add answers where the text can be exactly recovered from the
                # document. If this CAN'T happen it's likely due to weird Unicode
                # stuff so we will just skip the example.
                #
                # Note that this means for training mode, every example is NOT
                # guaranteed to be preserved.
                actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = " ".join(
                    whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    logger.warning("Could not find answer: '%s' vs. '%s'",
                                   actual_text, cleaned_answer_text)
            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ""
                
        example = QuacExample(
                    example_id=example_id,
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                    followup=entry['followup'], 
                    yesno=entry['yesno'], 
                    retrieval_label=int(entry['retrieval_label']), 
                    history=history)
        
        feature = convert_example_to_feature(example, self._tokenizer, is_training=self._is_training)
        
        # when evaluating, we save all examples and features
        # so that we can recover answer texts
        if not self._is_training:
            self.all_examples[example_id] = example
            self.all_features[example_id] = feature
            
        
        if self._is_training:      
            return {'input_ids': np.asarray(feature.input_ids), 
                    'segment_ids': np.asarray(feature.segment_ids), 
                    'input_mask': np.asarray(feature.input_mask), 
                    # 'cls_index': feature.cls_index, 
                    # 'p_mask': feature.p_mask, 
                    'start_position': feature.start_position, 
                    'end_position': feature.end_position, 
                    'retrieval_label': feature.retrieval_label}
        else:
            return {'input_ids': np.asarray(feature.input_ids), 
                    'segment_ids': np.asarray(feature.segment_ids), 
                    'input_mask': np.asarray(feature.input_mask), 
                    # 'cls_index': feature.cls_index, 
                    # 'p_mask': feature.p_mask, 
                    'example_id': feature.example_id}
        

    def __len__(self):
        return self._total_data


    
class LazyQuacDatasetGlobal(LazyQuacDataset):
    # when the global mode is on, we ignore the weak labels for answers
    # and maxmize the prob of the true answer given all passages
    def __getitem__(self, idx):
        """read a line of preprocessed open-retrieval quac file into a quac example"""
        line = linecache.getline(self._filename, idx + 1)    
        entry = json.loads(line.strip())
        qas_id = entry["qid"]
        orig_question_text = entry["question"]
        retrieval_labels = entry['retrieval_labels']
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
        
        batch = []
        paragraph_texts = entry['evidences']
        for i, (paragraph_text, retrieval_label) in enumerate(zip(paragraph_texts, retrieval_labels)):
            example_id = '{}_{}'.format(qas_id, i)
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)            

            start_position = None
            end_position = None
            orig_answer_text = None
            is_impossible = False        
            
            
            if self._is_training:                
                if entry['answer']['text'] in ['CANNOTANSWER', 'NOTRECOVERED'] or retrieval_label == 0:
                    is_impossible = True

                if not is_impossible:
                    answer = entry['answer']
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.warning("Could not find answer: '%s' vs. '%s'",
                                       actual_text, cleaned_answer_text)
                else:
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""

            example = QuacExample(
                        example_id=example_id,
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible,
                        followup=entry['followup'], 
                        yesno=entry['yesno'], 
                        retrieval_label=retrieval_label, 
                        history=history)

            feature = convert_example_to_feature(example, self._tokenizer, is_training=self._is_training)               

            # when evaluating, we save all examples and features
            # so that we can recover answer texts
            if not self._is_training:
                self.all_examples[example_id] = example
                self.all_features[example_id] = feature

            
            if self._is_training:
                if retrieval_label:
                    batch_feature = {'input_ids': np.asarray(feature.input_ids), 
                                    'segment_ids': np.asarray(feature.segment_ids), 
                                    'input_mask': np.asarray(feature.input_mask), 
                                    # 'cls_index': feature.cls_index, 
                                    # 'p_mask': feature.p_mask, 
                                     # the true passge might be at any position
                                    'start_position': feature.start_position + i * self._max_seq_length, 
                                    'end_position': feature.end_position + i * self._max_seq_length, 
                                    'retrieval_label': feature.retrieval_label}
                    
                    
                else:
                    batch_feature = {'input_ids': np.asarray(feature.input_ids), 
                                    'segment_ids': np.asarray(feature.segment_ids), 
                                    'input_mask': np.asarray(feature.input_mask), 
                                    # 'cls_index': feature.cls_index, 
                                    # 'p_mask': feature.p_mask, 
                                    'start_position': -1, 
                                    'end_position': -1, 
                                    'retrieval_label': feature.retrieval_label}
            else:
                batch_feature = {'input_ids': np.asarray(feature.input_ids), 
                                'segment_ids': np.asarray(feature.segment_ids), 
                                'input_mask': np.asarray(feature.input_mask), 
                                # 'cls_index': feature.cls_index, 
                                # 'p_mask': feature.p_mask, 
                                'example_id': feature.example_id}
            
            batch.append(batch_feature)
        
        collated = {}
        
        keys = batch[0].keys()
        for key in keys:
            if key != 'example_id':
                collated[key] = np.vstack([dic[key] for dic in batch])
        if 'example_id' in keys:
            collated['example_id'] = [dic['example_id'] for dic in batch]
        # print(collated)
        # print(collated['input_ids'].shape)
        return collated



    
def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False
    

def convert_example_to_feature(example, tokenizer, max_seq_length=512,
                                 doc_stride=384, max_query_length=125, is_training=True,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 sequence_a_is_doc=False):
    """Convert a single QuacExample to features (model input)"""

    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[-max_query_length:]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training and example.is_impossible:
        tok_start_position = -1
        tok_end_position = -1
    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
    assert max_tokens_for_doc >= 384, max_tokens_for_doc

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    
    # we set the doc_stride to 384, which is the max length of evidence text,
    # meaning that each evidence has exactly one _DocSpan
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)
    assert len(doc_spans) == 1, (max_tokens_for_doc, example)
    # if len(doc_spans) > 1:
        # print(len(doc_spans), example)
    #     doc_spans = [doc_spans[0]]

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = []

        # CLS token at the beginning
        if not cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            p_mask.append(0)
            cls_index = 0

        # XLNet: P SEP Q SEP CLS
        # Others: CLS Q SEP P SEP
        if not sequence_a_is_doc:
            # Query
            tokens += query_tokens
            segment_ids += [sequence_a_segment_id] * len(query_tokens)
            p_mask += [1] * len(query_tokens)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

        # Paragraph
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            if not sequence_a_is_doc:
                segment_ids.append(sequence_b_segment_id)
            else:
                segment_ids.append(sequence_a_segment_id)
            p_mask.append(0)
        paragraph_len = doc_span.length

        if sequence_a_is_doc:
            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            tokens += query_tokens
            segment_ids += [sequence_b_segment_id] * len(query_tokens)
            p_mask += [1] * len(query_tokens)

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_b_segment_id)
        p_mask.append(1)

        # CLS token at the end
        if cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            p_mask.append(0)
            cls_index = len(tokens) - 1  # Index of classification token

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_token)
            input_mask.append(0 if mask_padding_with_zero else 1)
            segment_ids.append(pad_token_segment_id)
            p_mask.append(1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        span_is_impossible = example.is_impossible
        start_position = None
        end_position = None
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
                span_is_impossible = True
            else:
                if sequence_a_is_doc:
                    doc_offset = 0
                else:
                    doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        if is_training and span_is_impossible:
            start_position = cls_index
            end_position = cls_index

        if False:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.example_id))
            logger.info("example_id: %s" % (example.example_id))
            logger.info("qid of the example: %s" % (example.qas_id))
            logger.info("doc_span_index: %s" % (doc_span_index))
            logger.info("tokens: %s" % " ".join(tokens))
            logger.info("token_to_orig_map: %s" % " ".join([
                "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
            logger.info("token_is_max_context: %s" % " ".join([
                "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
            ]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if is_training and span_is_impossible:
                logger.info("impossible example")
            if is_training and not span_is_impossible:
                answer_text = " ".join(tokens[start_position:(end_position + 1)])
                logger.info("start_position: %d" % (start_position))
                logger.info("end_position: %d" % (end_position))
                logger.info("retrieval_label: %d" % (example.retrieval_label))
                logger.info(
                    "answer: %s" % (answer_text))

        feature = InputFeatures(
                    unique_id=example.example_id,
                    example_id=example.example_id,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible, 
                    retrieval_label=example.retrieval_label)

    return feature


def gen_reader_features(qids, question_texts, answer_texts, answer_starts, passage_ids,
                        passages, all_retrieval_labels, reader_tokenizer, max_seq_length, is_training=False):
    # print('all_retrieval_labels', all_retrieval_labels, type(all_retrieval_labels))
    batch_features = []
    all_examples, all_features = {}, {}
    for (qas_id, question_text, answer_text, answer_start, pids_per_query,
         paragraph_texts, retrieval_labels) in zip(qids, question_texts, answer_texts, answer_starts, 
                                                   passage_ids, passages, all_retrieval_labels):
        # print('retrieval_labels', retrieval_labels)
        per_query_features = []
        for i, (pid, paragraph_text, retrieval_label) in enumerate(zip(pids_per_query, paragraph_texts, retrieval_labels)):
            # print('retrieval_label', retrieval_label)
            example_id = f'{qas_id}*{pid}'
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            start_position = None
            end_position = None
            orig_answer_text = None
            is_impossible = False

            if is_training:
                if answer_text in ['CANNOTANSWER', 'NOTRECOVERED'] or retrieval_label == 0:
                    is_impossible = True

                if not is_impossible:
                    orig_answer_text = answer_text
                    answer_offset = answer_start
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset +
                                                       answer_length - 1]
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(
                        doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.warning("Could not find answer: '%s' vs. '%s'",
                                       actual_text, cleaned_answer_text)
                else:
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""

            example = QuacExample(
                example_id=example_id,
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                is_impossible=is_impossible,
                retrieval_label=retrieval_label)

            feature = convert_example_to_feature(
                example, reader_tokenizer, is_training=is_training)

            # when evaluating, we save all examples and features
            # so that we can recover answer texts
            if not is_training:
                all_examples[example_id] = example
                all_features[example_id] = feature

            if is_training:
                if retrieval_label:
                    per_query_feature = {'input_ids': np.asarray(feature.input_ids),
                                     'segment_ids': np.asarray(feature.segment_ids),
                                     'input_mask': np.asarray(feature.input_mask),
                                     # 'cls_index': feature.cls_index,
                                     # 'p_mask': feature.p_mask,
                                     # the true passge might be at any position
                                     'start_position': feature.start_position + i * max_seq_length,
                                     'end_position': feature.end_position + i * max_seq_length,
                                     'retrieval_label': feature.retrieval_label}

                else:
                    per_query_feature = {'input_ids': np.asarray(feature.input_ids),
                                     'segment_ids': np.asarray(feature.segment_ids),
                                     'input_mask': np.asarray(feature.input_mask),
                                     # 'cls_index': feature.cls_index,
                                     # 'p_mask': feature.p_mask,
                                     'start_position': -1,
                                     'end_position': -1,
                                     'retrieval_label': feature.retrieval_label}
            else:
                per_query_feature = {'input_ids': np.asarray(feature.input_ids),
                                 'segment_ids': np.asarray(feature.segment_ids),
                                 'input_mask': np.asarray(feature.input_mask),
                                 # 'cls_index': feature.cls_index,
                                 # 'p_mask': feature.p_mask,
                                 'example_id': feature.example_id}

            per_query_features.append(per_query_feature)

        collated = {}

        keys = per_query_features[0].keys()
        for key in keys:
            if key != 'example_id':
                collated[key] = np.vstack([dic[key] for dic in per_query_features])
        if 'example_id' in keys:
            collated['example_id'] = [dic['example_id'] for dic in per_query_features]
        # print('collated', collated)
        # print(collated['input_ids'].shape)
        batch_features.append(collated)
    
    # print('batch_features', batch_features)
    batch = {}
    keys = batch_features[0].keys()
    for key in keys:
        if key != 'example_id':
            batch[key] = np.stack([dic[key] for dic in batch_features], axis=0)
            batch[key] = torch.from_numpy(batch[key])
    if 'example_id' in keys:
        batch['example_id'] = []
        for item in batch_features:
            batch['example_id'].extend(item['example_id'])
    # print('batch', batch)
    if is_training:
        return batch
    else:
        return batch, all_examples, all_features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

# the retrieval_logits is for reranking
RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits", 'retrieval_logits', 'retriever_prob'], 
                                   defaults=(None,)*4+ (1.0,))

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    # example_id_to_features = collections.defaultdict(list)
    # for feature in all_features:
    #     example_id_to_features[feature.example_id].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", 'retrieval_logit', 'retriever_prob'])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    # for (example_id, example) in enumerate(all_examples):
    for example_id, example in all_examples.items():
        # features = example_id_to_features[example_id]
        feature = all_features[example_id]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        # for (feature_index, feature) in enumerate(features):
        result = unique_id_to_result[feature.unique_id]
        start_indexes = _get_best_indexes(result.start_logits, n_best_size)
        end_indexes = _get_best_indexes(result.end_logits, n_best_size)
        # if we could have irrelevant answers, get the min score of irrelevant
        if version_2_with_negative:
            feature_null_score = result.start_logits[0] + result.end_logits[0]
            if feature_null_score < score_null:
                score_null = feature_null_score
                min_null_feature_index = 0
                null_start_logit = result.start_logits[0]
                null_end_logit = result.end_logits[0]
                null_retrieval_logit = result.retrieval_logits[-1]
                null_retriever_prob = result.retriever_prob
        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=0,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=result.start_logits[start_index],
                        end_logit=result.end_logits[end_index],
                        retrieval_logit=result.retrieval_logits[-1],
                        retriever_prob=result.retriever_prob))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=0, # min_null_feature_index
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit, 
                    retrieval_logit=null_retrieval_logit, 
                    retriever_prob=null_retriever_prob))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", 'retrieval_logit', 'retriever_prob'])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            # feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = "CANNOTANSWER"
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit, 
                    retrieval_logit=pred.retrieval_logit, 
                    retriever_prob=pred.retriever_prob))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "CANNOTANSWER" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="CANNOTANSWER",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit, 
                        retrieval_logit=null_retrieval_logit,
                        retriever_prob=null_retriever_prob))
                
            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest)==1:
                nbest.insert(0,
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, 
                                     retrieval_logit=0.0, retriever_prob=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, retrieval_logit=0.0, retriever_prob=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text != 'CANNOTANSWER':
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output['retrieval_logit'] = entry.retrieval_logit
            output['retriever_prob'] = entry.retriever_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            # all_predictions[example.qas_id] = nbest_json[0]["text"]
            all_predictions[example.example_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = (score_null - 
                          best_non_null_entry.start_logit - 
                          best_non_null_entry.end_logit)
            scores_diff_json[example.example_id] = score_diff
            if score_diff > null_score_diff_threshold:
                example_prediction = {'text': 'CANNOTANSWER',
                                      'start_logit': null_start_logit,
                                      'end_logit': null_end_logit,
                                      'retrieval_logit': null_retrieval_logit, 
                                      'retriever_prob': null_retriever_prob,
                                      'example_id': example.example_id}
            else:
                example_prediction = {'text': best_non_null_entry.text, 
                                      'start_logit': best_non_null_entry.start_logit,
                                      'end_logit': best_non_null_entry.end_logit,
                                      'retrieval_logit': best_non_null_entry.retrieval_logit, 
                                      'retriever_prob': best_non_null_entry.retriever_prob,
                                      'example_id': example.example_id}
                    
        if example.qas_id in all_predictions:            
            all_predictions[example.qas_id].append(example_prediction)
        else:
            all_predictions[example.qas_id] = [example_prediction]
            
        all_nbest_json[example.example_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions

def write_final_predictions(all_predictions, final_prediction_file, 
                            use_rerank_prob=True, use_retriever_prob=False):
    """convert instance level predictions to quac predictions"""
    logger.info("Writing final predictions to: %s" % (final_prediction_file))
    
    turn_level_preds = {}
    for qid, preds in all_predictions.items():
        qa_scores = []
        rerank_scores = []
        for pred in preds:
            qa_scores.append(pred['start_logit'] + pred['end_logit'])
            rerank_scores.append(pred['retrieval_logit'])
        qa_probs = np.asarray(_compute_softmax(qa_scores))
        rerank_probs = np.asarray(_compute_softmax(rerank_scores))
        
        total_scores = qa_probs
        if use_rerank_prob:
            total_scores = total_scores * rerank_probs
        if use_retriever_prob:
            total_scores = total_scores * pred['retriever_prob']
      
        best_idx = np.argmax(total_scores)
        turn_level_preds[qid] = preds[best_idx]
        
    dialog_level_preds = {}
    for qid, pred in turn_level_preds.items():
        dialog_id = qid.split('#')[0]
        if dialog_id in dialog_level_preds:
            dialog_level_preds[dialog_id]['best_span_str'].append(pred['text'])
            dialog_level_preds[dialog_id]['qid'].append(qid)
            dialog_level_preds[dialog_id]['yesno'].append('x')
            dialog_level_preds[dialog_id]['followup'].append('y')
        else:
            dialog_level_preds[dialog_id] = {}
            dialog_level_preds[dialog_id]['best_span_str'] = [pred['text']]
            dialog_level_preds[dialog_id]['qid'] = [qid]
            dialog_level_preds[dialog_id]['yesno'] = ['x']
            dialog_level_preds[dialog_id]['followup'] = ['y']
            
    with open(final_prediction_file, 'w') as fout:
        for pred in dialog_level_preds.values():
            fout.write(json.dumps(pred) + '\n')
            
    return dialog_level_preds.values()

# For XLNet (and XLM which uses the same head)
RawResultExtended = collections.namedtuple("RawResultExtended",
    ["unique_id", "start_top_log_probs", "start_top_index",
     "end_top_log_probs", "end_top_index", "cls_logits"])


def write_predictions_extended(all_examples, all_features, all_results, n_best_size,
                                max_answer_length, output_prediction_file,
                                output_nbest_file,
                                output_null_log_odds_file, orig_data_file,
                                start_n_top, end_n_top, version_2_with_negative,
                                tokenizer, verbose_logging):
    """ XLNet write prediction logic (more complex than Bert's).
        Write final predictions to the json file and log-odds of null if needed.
        Requires utils_squad_evaluate.py
    """
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index",
        "start_log_prob", "end_log_prob"])

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

    logger.info("Writing predictions to: %s", output_prediction_file)
    # logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_id_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_id_to_features[feature.example_id].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True)

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            # XLNet un-tokenizer
            # Let's keep it simple for now and see if we need all this later.
            # 
            # tok_start_to_orig_index = feature.tok_start_to_orig_index
            # tok_end_to_orig_index = feature.tok_end_to_orig_index
            # start_orig_pos = tok_start_to_orig_index[pred.start_index]
            # end_orig_pos = tok_end_to_orig_index[pred.end_index]
            # paragraph_text = example.paragraph_text
            # final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

            # Previously used Bert untokenizer
            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, tokenizer.do_lower_case,
                                        verbose_logging)

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="", start_log_prob=-1e6,
                end_log_prob=-1e6))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        probs = _compute_softmax(total_scores) # the prob of being the best answer among nbest answers

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_log_prob"] = entry.start_log_prob
            output["end_log_prob"] = entry.end_log_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        assert best_non_null_entry is not None

        score_diff = score_null
        scores_diff_json[example.qas_id] = score_diff
        # note(zhiliny): always predict best_non_null_entry
        # and the evaluation script will search for the best threshold
        all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    with open(orig_data_file, "r", encoding='utf-8') as reader:
        orig_data = json.load(reader)["data"]

    qid_to_has_ans = make_qid_to_has_ans(orig_data)
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact_raw, f1_raw = get_raw_scores(orig_data, all_predictions)
    out_eval = {}

    find_all_best_thresh_v2(out_eval, all_predictions, exact_raw, f1_raw, scores_diff_json, qid_to_has_ans)

    return out_eval


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def get_retrieval_metrics(evaluator, all_predictions, eval_retriever_probs=False):
    rerank_run = {}
    for qid, preds in all_predictions.items():
        rerank_run[qid] = {}
        for pred in preds:
            pid = pred['example_id'].split('*')[1] if eval_retriever_probs else pred['example_id']
            rerank_run[qid][pid] = pred['retrieval_logit']
    # print('rerank_run', rerank_run)
    rerank_metrics = evaluator.evaluate(rerank_run)
    rerank_mrr_list = [v['recip_rank'] for v in rerank_metrics.values()]
    rerank_recall_list = [v['recall_5'] for v in rerank_metrics.values()]    
    return_dict = {'rerank_mrr': np.average(rerank_mrr_list), 'rerank_recall': np.average(rerank_recall_list)}
    
    if eval_retriever_probs:
        retriever_run = {}
        for qid, preds in all_predictions.items():
            retriever_run[qid] = {}
            for pred in preds:
                pid = pred['example_id'].split('*')[1]
                retriever_run[qid][pid] = pred['retriever_prob']
        # print('retriever_run', retriever_run)
        retriever_metrics = evaluator.evaluate(retriever_run)
        retriever_mrr_list = [v['recip_rank'] for v in retriever_metrics.values()]
        retriever_recall_list = [v['recall_5'] for v in retriever_metrics.values()]    
        return_dict.update({'retriever_mrr': np.average(retriever_mrr_list), 
                            'retriever_recall': np.average(retriever_recall_list)})
        
    return return_dict
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    