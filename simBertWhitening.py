# coding: utf-8
from pathlib import Path
import os

import torch
import numpy as np
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer

from utils import mean_pooling


class SimBertWhitening(nn.Module):
    def __init__(self, model_path: Path, model_name: str):
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError('model not found.')
        if model_name in ['bert_base_chinese', 'roformer', 'bert_uer_large']:
            self.bert = BertModel.from_pretrained(model_path)
        else:
            self.bert = AutoModel.from_pretrained(model_path)

    @staticmethod
    def whitening(vecs: torch.FloatTensor, n_components):
        """计算kernel和bias
        最后的变换：y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, _ = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return torch.FloatTensor(W[:, :n_components]), -mu

    @staticmethod
    def transformAndNnormalize(vecs: torch.FloatTensor, kernel: torch.FloatTensor = None, bias: torch.FloatTensor = None) -> torch.FloatTensor:
        """应用变换，然后标准化
        """
        if not (kernel is None or bias is None):
            vecs = torch.matmul((vecs + bias), kernel)
        return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5

    def forward(self, input_ids, token_type_ids, attention_mask, vec_type: str):
        output = self.bert(
                            input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        if vec_type == 'last':
            return mean_pooling(output.hidden_states[-1], attention_mask)
        elif vec_type == 'pooler':
            return output.pooler_output
        elif vec_type == 'last2avg':
            return mean_pooling(output.hidden_states[-2] + output.hidden_states[-1], attention_mask)
        elif vec_type == 'first_last_avg':
            return mean_pooling(output.hidden_states[1] + output.hidden_states[-1], attention_mask)
        elif vec_type == 'cls':
            return output.hidden_states[-1][:, 0, :]
        else:
            raise TypeError('vec_type error')


class Tokenizer(nn.Module):
    def __init__(self, vocab_path: Path, model_name: str):
        super().__init__()
        if not os.path.exists(vocab_path):
            raise FileNotFoundError('vocab not found.')
        if model_name in ['bert_base_chinese', 'bert_uer_large']:
            self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_path)

    def forward(self, sentence: str, max_length: int):
        encodes = self.tokenizer.encode_plus(
                                            sentence,
                                            padding='max_length',
                                            truncation='only_first',
                                            max_length=max_length,
                                            return_tensors='pt'
                                            )
        return encodes
