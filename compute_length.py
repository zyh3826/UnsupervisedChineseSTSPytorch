# coding: utf-8
from pathlib import Path
import os

from dataloader import DataProcessor
from utils import DATASET_ROOT_PATH


def computeLength(data_path: Path, dataset: str):
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
    length_a, length_b, sum_a, sum_b = 0, 0, 0, 0
    for _, texts in data.items():
        data_a, data_b, _ = texts
        sum_a += sum(map(len, data_a))
        sum_b += sum(map(len, data_b))
        length_a += len(data_a)
        length_b += len(data_b)
    info = 'Dataset: {}, length_a: {}, length_b: {}'
    print(info.format(dataset, sum_a // length_a, sum_b // length_b))
    print('\n\n')


def main():
    datasets = ['LCQMC', 'ChineseTextualInference', 'CCKS_2018_3', 'BQ_corpus', 'ATEC_CCKS', 'ATEC', 'STS_B', 'ChineseSTS', 'cnsd_mnli', 'cnsd_snli']
    for d in datasets:
        path = os.path.join(DATASET_ROOT_PATH, d)
        computeLength(path, d)


if __name__ == '__main__':
    import fire
    fire.Fire()
