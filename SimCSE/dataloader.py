# coding: utf-8
from typing import List, Tuple
from pathlib import Path
import json
import numbers
import torch

from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import RoFormerTokenizer

pattern = u'[^\u4e00-\u9fa50-9a-zA-Z]+'


class DataProcessor(object):

    @staticmethod
    def read_LCQMC(path: Path) -> Tuple[List[str], List[str], List[str]]:
        f = open(path, 'r', encoding='utf-8')
        data_a, data_b, labels = [], [], []
        for line in tqdm(f, desc='Loading LCQMC data'):
            seq_a, seq_b, label = line.strip('\n').split('\t')
            data_a.append(seq_a)
            data_b.append(seq_b)
            labels.append(int(label))
        return (data_a, data_b, labels)

    @staticmethod
    def read_ChineseTextualInference(path: Path) -> Tuple[List[str], List[str], List[str]]:
        label2index = {'neutral': 0.5, 'contradiction': 0, 'entailment': 1}
        data_a, data_b, labels = [], [], []
        f = open(path, 'r', encoding='utf-8')
        for line in tqdm(f, desc='Loading ChineseTextualInference data'):
            try:
                seq_a, seq_b, label = line.strip('\n').split('\t')
            except ValueError:
                continue
            data_a.append(seq_a)
            data_b.append(seq_b)
            labels.append(label2index[label])
        return (data_a, data_b, labels)

    @staticmethod
    def read_BQ_corpus(path: Path) -> Tuple[List[str], List[str], List[str]]:
        print('Loading BQ_corpus')
        f = pd.read_csv(path)
        f['label'] = f['label'].apply(lambda x: int(x))
        return f['sentence1'].to_list(), f['sentence2'].to_list(), f['label'].to_list()

    @staticmethod
    def read_ATEC(path: Path) -> Tuple[List[str], List[str], List[str]]:
        data_a, data_b, labels = [], [], []
        f = open(path, 'r', encoding='utf-8')
        for line in tqdm(f, desc='Loading ATEC data'):
            try:
                _, seq_a, seq_b, label = line.strip('\n').split('\t')
            except ValueError:
                continue
            data_a.append(seq_a)
            data_b.append(seq_b)
            labels.append(int(label))
        return (data_a, data_b, labels)

    @staticmethod
    def read_STS_B(path: Path) -> Tuple[List[str], List[str], List[str]]:
        data_a, data_b, labels = [], [], []
        f = open(path, 'r', encoding='utf-8')
        for line in tqdm(f, desc='Loading STS_B data'):
            try:
                seq_a, seq_b, label = line.strip('\n').split('\t')
            except ValueError:
                continue
            data_a.append(seq_a)
            data_b.append(seq_b)
            labels.append(int(label))
        return (data_a, data_b, labels)

    @staticmethod
    def read_ChineseSTS(path: Path) -> Tuple[List[str], List[str], List[str]]:
        data_a, data_b, labels = [], [], []
        f = open(path, 'r', encoding='utf-8')
        for line in tqdm(f, desc='Loading ChineseSTS data'):
            try:
                content = line.strip('\n').split('\t')
                seq_a = content[1]
                seq_b = content[3]
                label = content[-1]
            except ValueError:
                continue
            data_a.append(seq_a)
            data_b.append(seq_b)
            labels.append(float(label))
        return (data_a, data_b, labels)

    @staticmethod
    def read_cnsd_mnli(path: Path) -> Tuple[List[str], List[str], List[str]]:
        label2index = {'neutral': 0.5, 'contradiction': 0, 'entailment': 1}
        data_a, data_b, labels = [], [], []
        f = open(path, 'r', encoding='utf-8')
        for line in tqdm(f, desc='Loading cnsd_mnli data'):
            j_line = json.loads(line.strip('\n'))
            try:
                labels.append(label2index[j_line['gold_label']])
            except KeyError:
                continue
            data_a.append(j_line['sentence1'])
            data_b.append(j_line['sentence2'])
        return (data_a, data_b, labels)

    @staticmethod
    def read_cnsd_snli(path: Path) -> Tuple[List[str], List[str], List[str]]:
        label2index = {'neutral': 0.5, 'contradiction': 0, 'entailment': 1}
        data_a, data_b, labels = [], [], []
        f = open(path, 'r', encoding='utf-8')
        for line in tqdm(f, desc='Loading cnsd_snli data', ncols=200):
            j_line = json.loads(line.strip('\n'))
            try:
                labels.append(label2index[j_line['gold_label']])
            except KeyError:
                continue
            data_a.append(j_line['sentence1'])
            data_b.append(j_line['sentence2'])
        return (data_a, data_b, labels)


class MyDataset(Dataset):
    def __init__(self, name: str, data_path: str) -> None:
        super().__init__()
        if name == 'LCQMC':
            data = DataProcessor.read_LCQMC(data_path)
        elif name == 'ChineseTextualInference':
            data = DataProcessor.read_ChineseTextualInference(data_path)
        elif name == 'BQ_corpus':
            data = DataProcessor.read_BQ_corpus(data_path)
        elif name == 'ATEC':
            data = DataProcessor.read_ATEC(data_path)
        elif name == 'STS_B':
            data = DataProcessor.read_STS_B(data_path)
        elif name == 'ChineseSTS':
            data = DataProcessor.read_ChineseSTS(data_path)
        elif name == 'cnsd_mnli':
            data = DataProcessor.read_cnsd_mnli(data_path)
        elif name == 'cnsd_snli':
            data = DataProcessor.read_cnsd_snli(data_path)
        else:
            raise TypeError('No such dataset: {}'.format(name))
        self.data_a, self.data_b, self.labels = data
        assert len(self.data_a) == len(self.data_b) == len(self.labels), 'Length not equal'

    def __getitem__(self, index):
        if isinstance(index, numbers.Integral):
            return self.data_a[index], self.data_b[index], self.labels[index]
        elif isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step
            return self.data_a[start: stop: step], self.data_b[start: stop: step], self.labels[start: stop: step]
        else:
            raise TypeError('index type error.')

    def __len__(self):
        return len(self.data_a)


class MyDataLoader:
    def __init__(self, max_len: int, batch_size: int, vocab_path: str, device: str) -> None:
        self.max_len = max_len
        self.batch_size = batch_size
        if 'roformer' in vocab_path:
            self.tokenizer = RoFormerTokenizer.from_pretrained(vocab_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        self.datasets = {}
        self.device = device

    def load(self, name: str, data_path: str, shuffle: bool = False) -> DataLoader:
        if name in self.datasets:
            dataset = self.datasets[name]
        else:
            dataset = MyDataset(name, data_path)
            self.datasets[name] = dataset
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    def collate_fn(self, examples):
        sentences_a = [item[0] for item in examples]
        sentences_b = [item[1] for item in examples]
        labels = [item[2] for item in examples]
        max_length = max(map(len, sentences_a))
        if max_length > self.max_len:
            max_length = self.max_len
        encodes_a = self.tokenizer(sentences_a, padding='max_length', truncation='longest_first', max_length=max_length, return_tensors='pt')
        for key in encodes_a:
            encodes_a[key] = encodes_a[key].to(self.device)
        encodes_b = self.tokenizer(sentences_b, padding='max_length', truncation='longest_first', max_length=max_length, return_tensors='pt')
        for key in encodes_b:
            encodes_b[key] = encodes_b[key].to(self.device)
        labels = torch.FloatTensor(labels)
        return encodes_a, encodes_b, labels


if __name__ == '__main__':
    vocab_path = '/source/c0/NLPSource/embedding/transformer_based/chinese_roformer-sim-char-ft_L-12_H-768_A-12'
    data_loader = MyDataLoader(64, 3, vocab_path=vocab_path, device='cpu')
    name = 'STS_B'
    data_path = '/source/c0/NLPSource/Datasets/STS/STS_B/mini_test.txt'
    train = data_loader.load(name, data_path)
    for batch in train:
        _ = batch
