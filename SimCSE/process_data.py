# coding: utf-8
import os
import random
from glob import glob

from tqdm import tqdm

from dataloader import DataProcessor
from utils import DATASET_ROOT_PATH, DATASETS

random.seed(13)

for dataset_name in tqdm(DATASETS, desc='Extracting data'):
    root = os.path.join(DATASET_ROOT_PATH, dataset_name)
    func_name = 'read_' + dataset_name
    func = getattr(DataProcessor, func_name)
    all_data = []
    for path in glob(os.path.join(root, '*.data')):
        data_a, data_b, labels = func(path)
        all_data += data_a
        all_data += data_b
    print('=======================all_data len: ', len(all_data))
    save = open(os.path.join(root, 'extract_data.txt'), 'w', encoding='utf-8')
    for line in all_data:
        save.write(line + '\n')
    save.close()
    random_10k = random.sample(all_data, 10000)
    save = open(os.path.join(root, 'random_10k.txt'), 'w', encoding='utf-8')
    for line in random_10k:
        save.write(line + '\n')
    save.close()
