# coding: utf-8
import glob
import os
from pathlib import Path
import re
from typing import Dict, List
import json
import copy

import torch
from torch.utils.data import TensorDataset
import pandas as pd
from tqdm import tqdm

pattern = u'[^\u4e00-\u9fa50-9a-zA-Z]+'


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid: int, text_a: str, text_b: str):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids: torch.LongTensor, token_type_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):

    def __init__(self, feature_a: InputFeature, feature_b: InputFeature):
        self.feature_a = feature_a
        self.feature_b = feature_b

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):

    @staticmethod
    def readLCQMC(path: Path) -> Dict[str, List[List[str]]]:
        texts = glob.glob(os.path.join(path, '*.txt'))
        data = {}
        for text in texts:
            data_a, data_b, labels = [], [], []
            name = text.split('/')[-1].split('.')[0]
            data[name] = []
            f = open(text, 'r', encoding='utf-8')
            for line in tqdm(f, desc='Loading data'):
                seq_a, seq_b, label = line.strip('\n').split('\t')
                data_a.append(re.sub(pattern, '', seq_a))
                data_b.append(re.sub(pattern, '', seq_b))
                labels.append(int(label))
            data[name].append(data_a)
            data[name].append(data_b)
            data[name].append(labels)
        return data

    @staticmethod
    def readChineseTextualInference(path: Path) -> Dict[str, List[List[str]]]:
        label2index = {'neutral': 0.5, 'contradiction': 0, 'entailment': 1}
        texts = glob.glob(os.path.join(path, '*.txt'))
        data = {}
        for text in texts:
            data_a, data_b, labels = [], [], []
            name = text.split('/')[-1].split('.')[0]
            data[name] = []
            f = open(text, 'r', encoding='utf-8')
            for line in tqdm(f, desc='Loading data'):
                try:
                    seq_a, seq_b, label = line.strip('\n').split('\t')
                except ValueError:
                    continue
                data_a.append(re.sub(pattern, '', seq_a))
                data_b.append(re.sub(pattern, '', seq_b))
                labels.append(label2index[label])
            data[name].append(data_a)
            data[name].append(data_b)
            data[name].append(labels)
        return data

    @staticmethod
    def readCCKS_2018_3(path: Path) -> Dict[str, List[List[str]]]:
        texts = glob.glob(os.path.join(path, '*.txt'))
        data = {}
        for text in texts:
            data_a, data_b, labels = [], [], []
            name = text.split('/')[-1].split('.')[0]
            data[name] = []
            f = open(text, 'r', encoding='utf-8')
            for line in tqdm(f, desc='Loading data'):
                try:
                    seq_a, seq_b, label = line.strip('\n').split('\t')
                except ValueError:
                    continue
                data_a.append(re.sub(pattern, '', seq_a))
                data_b.append(re.sub(pattern, '', seq_b))
                labels.append(int(label))
            data[name].append(data_a)
            data[name].append(data_b)
            data[name].append(labels)
        return data

    @staticmethod
    def readBQ_corpus(path: Path) -> Dict[str, List[List[str]]]:
        texts = glob.glob(os.path.join(path, '*.csv'))
        data = {}
        for text in texts:
            name = text.split('/')[-1].split('.')[0]
            data[name] = []
            f = pd.read_csv(text)
            f['sentence1'] = f['sentence1'].apply(lambda x: re.sub(pattern, '', x))
            f['sentence2'] = f['sentence2'].apply(lambda x: re.sub(pattern, '', x))
            f['label'] = f['label'].apply(lambda x: int(x))
            data[name].append(f['sentence1'].to_list())
            data[name].append(f['sentence2'].to_list())
            data[name].append(f['label'].to_list())
        return data

    @staticmethod
    def readATEC_CCKS(path: Path) -> Dict[str, List[List[str]]]:
        texts = glob.glob(os.path.join(path, '*.csv'))
        data = {}
        for text in texts:
            data_a, data_b, labels = [], [], []
            name = text.split('/')[-1].split('.')[0]
            data[name] = []
            f = open(text, 'r', encoding='utf-8')
            for line in tqdm(f, desc='Loading data'):
                try:
                    seq_a, seq_b, label = line.strip('\n').split('\t')
                except ValueError:
                    continue
                data_a.append(re.sub(pattern, '', seq_a))
                data_b.append(re.sub(pattern, '', seq_b))
                labels.append(int(label))
            data[name].append(data_a)
            data[name].append(data_b)
            data[name].append(labels)
        return data

    @staticmethod
    def readATEC(path: Path) -> Dict[str, List[List[str]]]:
        texts = glob.glob(os.path.join(path, '*.csv'))
        data = {}
        for text in texts:
            data_a, data_b, labels = [], [], []
            name = text.split('/')[-1].split('.')[0]
            data[name] = []
            f = open(text, 'r', encoding='utf-8')
            for line in tqdm(f, desc='Loading data'):
                try:
                    _, seq_a, seq_b, label = line.strip('\n').split('\t')
                except ValueError:
                    continue
                data_a.append(re.sub(pattern, '', seq_a))
                data_b.append(re.sub(pattern, '', seq_b))
                labels.append(int(label))
            data[name].append(data_a)
            data[name].append(data_b)
            data[name].append(labels)
        return data

    @staticmethod
    def readSTS_B(path: Path) -> Dict[str, List[List[str]]]:
        texts = glob.glob(os.path.join(path, '*.txt'))
        data = {}
        for text in texts:
            data_a, data_b, labels = [], [], []
            name = text.split('/')[-1].split('.')[0]
            data[name] = []
            f = open(text, 'r', encoding='utf-8')
            for line in tqdm(f, desc='Loading data'):
                try:
                    _, seq_a, seq_b, label = line.strip('\n').split('||')
                except ValueError:
                    continue
                data_a.append(re.sub(pattern, '', seq_a))
                data_b.append(re.sub(pattern, '', seq_b))
                labels.append(int(label))
            data[name].append(data_a)
            data[name].append(data_b)
            data[name].append(labels)
        return data

    @staticmethod
    def readChineseSTS(path: Path) -> Dict[str, List[List[str]]]:
        texts = glob.glob(os.path.join(path, '*.txt'))
        data = {}
        for text in texts:
            data_a, data_b, labels = [], [], []
            name = text.split('/')[-1].split('.')[0]
            data[name] = []
            f = open(text, 'r', encoding='utf-8')
            for line in tqdm(f, desc='Loading data'):
                try:
                    content = line.strip('\n').split('\t')
                    seq_a = content[1]
                    seq_b = content[3]
                    label = content[-1]
                except ValueError:
                    continue
                data_a.append(re.sub(pattern, '', seq_a))
                data_b.append(re.sub(pattern, '', seq_b))
                labels.append(float(label))
            data[name].append(data_a)
            data[name].append(data_b)
            data[name].append(labels)
        return data

    @staticmethod
    def readcnsd_mnli(path: Path) -> Dict[str, List[List[str]]]:
        label2index = {'neutral': 0.5, 'contradiction': 0, 'entailment': 1}
        texts = glob.glob(os.path.join(path, '*.jsonl'))
        data = {}
        for text in texts:
            data_a, data_b, labels = [], [], []
            name = text.split('/')[-1].split('.')[0]
            data[name] = []
            f = open(text, 'r', encoding='utf-8')
            for line in tqdm(f, desc='Loading data'):
                j_line = json.loads(line.strip('\n'))
                try:
                    labels.append(label2index[j_line['gold_label']])
                except KeyError:
                    continue
                data_a.append(re.sub(pattern, '', j_line['sentence1']))
                data_b.append(re.sub(pattern, '', j_line['sentence2']))
            data[name].append(data_a)
            data[name].append(data_b)
            data[name].append(labels)
        return data

    @staticmethod
    def readcnsd_snli(path: Path) -> Dict[str, List[List[str]]]:
        label2index = {'neutral': 0.5, 'contradiction': 0, 'entailment': 1}
        texts = glob.glob(os.path.join(path, '*.jsonl'))
        data = {}
        for text in texts:
            data_a, data_b, labels = [], [], []
            name = text.split('/')[-1].split('.')[0]
            data[name] = []
            f = open(text, 'r', encoding='utf-8')
            for line in tqdm(f, desc='Loading data', ncols=200):
                j_line = json.loads(line.strip('\n'))
                try:
                    labels.append(label2index[j_line['gold_label']])
                except KeyError:
                    continue
                data_a.append(re.sub(pattern, '', j_line['sentence1']))
                data_b.append(re.sub(pattern, '', j_line['sentence2']))
            data[name].append(data_a)
            data[name].append(data_b)
            data[name].append(labels)
        return data


def creatExamples(data_a: List[str], data_b: List[str]) -> List[InputExample]:
    if len(data_a) != len(data_b):
        raise TypeError('length bettween text_a and text_b is not equal')
    examples = []
    for idx, text in enumerate(tqdm(zip(data_a, data_b), desc='creat examples', total=len(data_a), ncols=200)):
        examples.append(InputExample(idx, text[0], text[1]))
    return examples


def convertExamplesToFeatures(examples: List[InputExample], tokenizer, length_a: int, length_b: int) -> List[InputFeatures]:
    features = []
    for example in tqdm(examples, desc='convert examples to features', ncols=200):
        encodes_a = tokenizer(example.text_a, length_a)
        encodes_b = tokenizer(example.text_b, length_b)
        feature_a = InputFeature(**encodes_a)
        feature_b = InputFeature(**encodes_b)
        features.append(InputFeatures(feature_a, feature_b))
    return features


def convertFeaturesToTensor(data_path: Path, dataset: str, tokenizer, length_a: int, length_b: int) -> Dict[str, TensorDataset]:
    if dataset == 'LCQMC':
        data = DataProcessor.readLCQMC(data_path)
    elif dataset == 'ChineseTextualInference':
        data = DataProcessor.readChineseTextualInference(data_path)
    elif dataset == 'CCKS_2018_3':
        data = DataProcessor.readCCKS_2018_3(data_path)
    elif dataset == 'BQ_corpus':
        data = DataProcessor.readBQ_corpus(data_path)
    elif dataset == 'ATEC_CCKS':
        data = DataProcessor.readATEC_CCKS(data_path)
    elif dataset == 'ATEC':
        data = DataProcessor.readATEC(data_path)
    elif dataset == 'STS_B':
        data = DataProcessor.readSTS_B(data_path)
    elif dataset == 'ChineseSTS':
        data = DataProcessor.readChineseSTS(data_path)
    elif dataset == 'cnsd_mnli':
        data = DataProcessor.readcnsd_mnli(data_path)
    elif dataset == 'cnsd_snli':
        data = DataProcessor.readcnsd_snli(data_path)
    else:
        raise TypeError('no such dataset')
    datasets = {}
    for name, texts in data.items():
        data_a, data_b, labels = texts
        examples = creatExamples(data_a, data_b)
        features = convertExamplesToFeatures(
            examples=examples,
            tokenizer=tokenizer,
            length_a=length_a,
            length_b=length_b)
        input_ids_a = torch.cat([item.feature_a.input_ids for item in features])
        token_type_ids_a = torch.cat([item.feature_a.token_type_ids for item in features])
        attention_mask_a = torch.cat([item.feature_a.attention_mask for item in features])

        input_ids_b = torch.cat([item.feature_b.input_ids for item in features])
        token_type_ids_b = torch.cat([item.feature_b.token_type_ids for item in features])
        attention_mask_b = torch.cat([item.feature_b.attention_mask for item in features])

        labels = torch.LongTensor(labels)
        datasets[name] = TensorDataset(input_ids_a, token_type_ids_a, attention_mask_a, input_ids_b, token_type_ids_b, attention_mask_b, labels)
    return datasets
