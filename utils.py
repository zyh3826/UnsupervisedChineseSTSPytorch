# coding: utf-8
from pathlib import Path
from typing import Tuple
import json

import torch
import scipy.stats as ss

DATASET_ROOT_PATH = '/source/d0/NLPSource/STSDatasets/'
MODEL_ROOT_PATH = '/source/d0/embedding/transformer_based'
WHITENING_SAVE_ROOT = '/source/d0/embedding/whitening'
RESULT_SAVE_PATH = '/source/d0/embedding/whitening/res128dim.json'

config = {
    'batch_size': 1024,
    'cuda_id': 1
}

DATASETS = {
    'STS_B': [18, 18],
    'LCQMC': [10, 10],
    'ChineseTextualInference': [23, 12],
    'BQ_corpus': [11, 11],
    'ATEC': [12, 12],
    'ChineseSTS': [21, 30],
    'cnsd_mnli': [32, 16],
    'cnsd_snli': [21, 10]
    # 'ATEC_CCKS': [11, 12],
    # 'CCKS_2018_3': [11, 11],
    }
VEC_TYPE = ['last', 'last2avg', 'first_last_avg', 'cls', 'pooler']
MODELS = {
    'simbert_L12_H768': 'chinese_simbert_L-12_H-768_A-12',
    'bert_base_chinese': 'bert_base_chinese',
    'hit_bert_wwm_ext': 'hit_bert_wwm_ext',
    'hit_bert_wwm_ext_large': 'hit_bert_wwm_ext_large',
    'wobert': 'chinese_wobert_plus_L-12_H-768_A-12',
    'roformer': 'chinese_roformer_L-12_H-768_A-12',
    'bert_uer_large': 'bert_uer_large',
    # 'sbert': 'stsb-xlm-r-multilingual',
}
OUTPUT_SIZE = {
    'simbert_L12_H768': 768,
    'bert_base_chinese': 768,
    'hit_bert_wwm_ext': 768,
    'hit_bert_wwm_ext_large': 1024,
    'wobert': 768,
    'roformer': 768,
    'bert_uer_large': 1024,
}


def saveWhitening(kernel: torch.Tensor, bias: torch.Tensor, path: Path):
    save = {
        'kernel': kernel,
        'bias': bias
    }
    torch.save(save, path)


def loadWhitening(path: Path) -> Tuple[torch.tensor, torch.tensor]:
    data = torch.load(path)
    kernel = data['kernel']
    bias = data['bias']
    return kernel, bias


class AttributeDict(dict):
    def __init__(self, arg=None):
        dict.__init__(self)
        if arg is not None:
            for k, v in arg.items():
                self[k] = v

    def __getattr__(self, name):
        if name in self:
            return self[name]
        return None

    def __setattr__(self, name, value):
        self[name] = value

    def __repr__(self):
        s = json.dumps(self, indent=2, ensure_ascii=False)
        return s

    def __str__(self):
        return self.__repr__()

    def get(self, name, default=None):
        if name in self:
            return self[name]
        if default is not None:
            return default
        raise AttributeError(name)

    def clone(self):
        return AttributeDict(dict(self))

    def remove(self, name):
        del self[name]


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def compute_corrcoef(x, y):
    """
    Spearman相关系数
    """
    return ss.spearmanr(x, y).correlation


g_config = AttributeDict(config)
