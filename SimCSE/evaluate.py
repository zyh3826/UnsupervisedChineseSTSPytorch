# coding: utf-8
import logging
import os
import time
import json
from collections import defaultdict
import sys

import scipy.stats as ss
import torch
from tqdm.std import tqdm
from transformers import BertConfig, HfArgumentParser, TrainingArguments
import numpy as np

from utils import (
                    MODEL_ROOT_PATH,
                    MODELS,
                    DATASET_ROOT_PATH,
                    DATASETS_PATH,
                    SAVE_ROOT)
from dataloader import MyDataLoader
from ToolFunction.SimCSE.args import ModelArguments, DataTrainingArguments
from ToolFunction.SimCSE.models import BertForCL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_corrcoef(x, y):
    """
    Spearman相关系数
    """
    return ss.spearmanr(x, y).correlation


def l2_normalize(vecs):
    """
        标准化
    """
    norms = (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def eval(**kwargs):
    logger.info('**********Begin STS Experiments**********')
    start = time.time()
    step = 0
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif kwargs:
        model_args, data_args, training_args = parser.parse_dict(kwargs)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    save_path = data_args.eval_result_save_path
    if os.path.exists(save_path):
        res = json.load(open(save_path, 'r', encoding='utf-8'))
        finished = set(res['model'])
    else:
        res = defaultdict(list)
        finished = set()
    for model_name in MODELS:
        if not os.path.isdir(os.path.join(SAVE_ROOT, model_name)):
            continue
        if model_name in finished:
            continue
        print('\n\n\n========================================')
        info = '**********Begining test model: {}**********'.format(model_name)
        logger.info(info)
        s = time.time()
        model_path = os.path.join(SAVE_ROOT, model_name)
        try:
            vocab_path = os.path.join(MODEL_ROOT_PATH, MODELS[model_name])
        except KeyError:
            continue
        dataloader = MyDataLoader(64, 256, vocab_path, device=training_args.device)
        model_name_or_path = os.path.join(MODEL_ROOT_PATH, MODELS[model_name])
        config = BertConfig.from_pretrained(model_name_or_path, hidden_dropout_prob=model_args.hidden_dropout_prob, **config_kwargs)
        for cp in os.listdir(model_path):
            if 'dimension' not in cp:
                dataset_name, pooler_type, sample = cp.split('-')[-3:]
                dimension = 768
                model_args.dimension_reduction = False
            else:
                dataset_name, pooler_type, sample, dimension_info = cp.split('-')[-4:]
                dimension = int(dimension_info.split('_')[-1])
                model_args.dimension_reduction = True
                model_args.output_size = dimension
            info = '**********Checkpoint: {}. Dataset: {}. Vector type: {}. Dimension: {}**********'
            logger.info(info.format(cp, dataset_name, pooler_type, dimension))
            dataset = dataloader.load(dataset_name, os.path.join(DATASET_ROOT_PATH, DATASETS_PATH[dataset_name]))
            checkpoint = os.path.join(model_path, cp)
            model_args.pooler_type = pooler_type
            model = BertForCL.from_pretrained(
                checkpoint,
                from_tf=False,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
            # model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin'), map_location='cpu'))
            model.to(training_args.device)
            model.eval()
            a_vec = torch.FloatTensor([])
            b_vec = torch.FloatTensor([])
            labels = torch.FloatTensor([])
            start_time = time.time()
            for batch in tqdm(dataset, desc='Testing'):
                with torch.no_grad():
                    encodes_a, encodes_b, ls = batch
                    outputs_a = model(**encodes_a, sent_emb=True, return_dict=True)
                    outputs_b = model(**encodes_b, sent_emb=True, return_dict=True)
                    a_vec = torch.cat((a_vec, outputs_a.pooler_output.cpu()), dim=0)
                    b_vec = torch.cat((b_vec, outputs_b.pooler_output.cpu()), dim=0)
                    labels = torch.cat((labels, ls), dim=0)
            a_vec = a_vec.numpy()
            b_vec = b_vec.numpy()
            labels = labels.numpy()
            logger.info('***** Computing spearman corrcoef *****')
            a_vec = l2_normalize(a_vec)
            b_vec = l2_normalize(b_vec)
            sims = (a_vec * b_vec).sum(axis=1)
            corrcoef = compute_corrcoef(labels, sims)
            res['model'].append(model_name)
            res['dataset'].append(dataset_name)
            res['pooler_type'].append(pooler_type)
            res['sample'].append(sample)
            res['dimension'].append(dimension)
            res['spearman'].append(corrcoef)
            step += 1
            if step % 10 == 0:
                logger.info('**********Saveing results**********')
                f = open(save_path, 'w', encoding='utf-8')
                json.dump(res, f, ensure_ascii=False, indent=4)
            info = '**********Checkpoint: {}. Dataset: {}. Vector type: {}. Dimension: {} finishing. Time use: {:.3f} s**********'
            logger.info(info.format(cp, dataset_name, pooler_type, dimension, time.time() - start_time))
        print('========================================')
        info = '**********Finishing test model: {}. Time use: {:.3f} min**********'
        logger.info(info.format(model_name, (time.time() - s) / 60))
        print('========================================')
        f = open(save_path, 'w', encoding='utf-8')
        json.dump(res, f, ensure_ascii=False, indent=4)
    logger.info('**********Finish STS Experiments. Time use: {:.3f} min**********'.format((time.time() - start) / 60))


def go():
    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    eval_result_save_path = os.path.join(SAVE_ROOT, 'simcse_results_with_dimension_reduction.json')
    kwargs = {'device': 'cuda:0', 'output_dir': output_dir, 'eval_result_save_path': eval_result_save_path, 'dataset_name': ''}
    eval(**kwargs)
    # 128 1.7
    # 256 1.16


if __name__ == '__main__':
    import fire
    fire.Fire(go)
