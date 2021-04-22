# coding: utf-8
import os
import time
from pathlib import Path
import logging
from typing import Mapping
import json

import torch
from torch.utils.data import DataLoader
import numpy as np

from simBertWhitening import SimBertWhitening, Tokenizer
from dataloader import convertFeaturesToTensor
from progressbar import ProgressBar
from utils import (
                    saveWhitening,
                    compute_corrcoef,
                    g_config,
                    MODEL_ROOT_PATH,
                    MODELS,
                    DATASET_ROOT_PATH,
                    DATASETS,
                    VEC_TYPE,
                    WHITENING_SAVE_ROOT,
                    RESULT_SAVE_PATH,
                    OUTPUT_SIZE)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def eval(
        model: SimBertWhitening,
        dataset_name: str,
        vec_type: str,
        n_components: int,
        whitening_save_path: Path,
        dataloaders: Mapping[str, DataLoader],
        whitening: bool):
    model.to(g_config.cuda_id)
    model.eval()
    all_names, all_weights, all_vecs, all_labels = [], [], [], []
    for name, dataloader in dataloaders.items():
        a_vec, b_vec, labels = torch.FloatTensor([]), torch.FloatTensor([]), torch.LongTensor([])
        all_names.append(name)
        all_weights.append(len(dataloader))
        pbar = ProgressBar(n_total=len(dataloader), desc='eval, dataset: {}, dataset type: {}'.format(dataset_name, name))
        print('\n')
        for step, batch in enumerate(dataloader):
            pbar(step)
            with torch.no_grad():
                input_ids_a, token_type_ids_a, attention_mask_a, input_ids_b, token_type_ids_b, attention_mask_b, batch_labels = batch
                tmp = {
                    'input_ids': input_ids_a.squeeze(1).cuda(g_config.cuda_id),
                    'token_type_ids': token_type_ids_a.squeeze(1).cuda(g_config.cuda_id),
                    'attention_mask': attention_mask_a.squeeze(1).cuda(g_config.cuda_id),
                    'vec_type': vec_type
                }
                a_output = model(**tmp)
                a_vec = torch.cat((a_vec, a_output.cpu()), dim=0)

                tmp = {
                    'input_ids': input_ids_b.squeeze(1).cuda(g_config.cuda_id),
                    'token_type_ids': token_type_ids_b.squeeze(1).cuda(g_config.cuda_id),
                    'attention_mask': attention_mask_b.squeeze(1).cuda(g_config.cuda_id),
                    'vec_type': vec_type
                }
                b_output = model(**tmp)
                b_vec = torch.cat((b_vec, b_output.cpu()), dim=0)
                labels = torch.cat((labels, batch_labels), dim=0)
        all_vecs.append((a_vec, b_vec))
        all_labels.append(labels)
    print('\n')
    if whitening:
        logger.info('Whitening')
        start = time.time()
        vecs = torch.cat([v for vecs in all_vecs for v in vecs], dim=0)
        kernel, bias = SimBertWhitening.whitening(vecs, n_components=n_components)
        saveWhitening(kernel, bias, whitening_save_path)
        logger.info('Finsih whitening. Time use: {:.3f} s'.format(time.time() - start))
    else:
        logger.info('No whitening')
        kernel = None
        bias = None
    all_corrcoefs = []
    for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
        a_vecs = SimBertWhitening.transformAndNnormalize(a_vecs, kernel, bias)
        b_vecs = SimBertWhitening.transformAndNnormalize(b_vecs, kernel, bias)
        sims = (a_vecs * b_vecs).sum(axis=1)
        corrcoef = compute_corrcoef(labels, sims)
        all_corrcoefs.append(corrcoef)
    all_corrcoefs.extend([
        np.average(all_corrcoefs),
        np.average(all_corrcoefs, weights=all_weights)
        ])
    logger.info('corrcoefs: ')
    res = {}
    for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
        res[name] = corrcoef
        print('%s: %s' % (name, corrcoef))
    return res


def main():
    logger.info('**********Begin STS Experiments**********')
    start = time.time()
    res = {}
    step = 0
    for model_name, model_path in MODELS.items():
        print('\n\n\n========================================')
        info = '**********Begining test model: {}**********'.format(model_name)
        logger.info(info)
        print('========================================')
        res[model_name] = {}
        s = time.time()
        model = SimBertWhitening(os.path.join(MODEL_ROOT_PATH, model_path), model_name=model_name)
        tokenizer = Tokenizer(os.path.join(MODEL_ROOT_PATH, model_path), model_name=model_name)
        for dataset_name, length in DATASETS.items():
            print('\n')
            info = '**********Loading dataset: {}. Tokenizer: {}**********'.format(dataset_name, model_name)
            logger.info(info)
            res[model_name][dataset_name] = {}
            dataset_path = os.path.join(DATASET_ROOT_PATH, dataset_name)
            length_a = length[0]
            length_b = length[1]
            datasets = convertFeaturesToTensor(dataset_path, dataset_name, tokenizer, length_a, length_b)
            dataloaders: Mapping[str, DataLoader] = {}
            for name, dataset in datasets.items():
                dataloaders[name] = DataLoader(dataset, batch_size=g_config.batch_size)
            info = '**********Loading dataset: {} success.**********'.format(dataset_name)
            logger.info(info)
            for v_type in VEC_TYPE:
                if model_name in ['wobert', 'roformer'] and v_type == 'pooler':
                    logger.info("model wobert and roformer don't have pooler")
                    continue
                res[model_name][dataset_name][v_type] = {}
                for whitening in [True, False]:
                    key = 'whitening' if whitening else 'not_whitening'
                    res[model_name][dataset_name][v_type][key] = {}
                    n_components = [128, 256, 384, 512, OUTPUT_SIZE[model_name]] if whitening else [OUTPUT_SIZE[model_name]]
                    for nc in n_components:
                        print('\n')
                        info = '**********Model: {}. Dataset: {}. Vector type: {}. Whitening: {}. n_components: {}**********'
                        logger.info(info.format(model_name, dataset_name, v_type, whitening, nc))
                        key = 'whitening' if whitening else 'not_whitening'
                        start_time = time.time()
                        tail = '{}-{}-{}-{}.bin'.format(model_name, dataset_name, v_type, nc)
                        whitening_save_path = os.path.join(WHITENING_SAVE_ROOT, model_name)
                        if not os.path.exists(whitening_save_path):
                            os.makedirs(whitening_save_path)
                        whitening_save_path = os.path.join(whitening_save_path, tail)
                        tmp = eval(
                                    model=model,
                                    dataset_name=dataset_name,
                                    vec_type=v_type,
                                    n_components=nc,
                                    whitening_save_path=whitening_save_path,
                                    dataloaders=dataloaders,
                                    whitening=whitening)
                        res[model_name][dataset_name][v_type][key][nc] = tmp
                        step += 1
                        if step % 10 == 0:
                            logger.info('**********Saveing results**********')
                            f = open(RESULT_SAVE_PATH, 'w', encoding='utf-8')
                            json.dump(res, f, ensure_ascii=False, indent=4)
                        info = '**********Model: {}. Dataset: {}. Vector type: {}. n_components: {} finishing. Whitening: {}. Time use: {:.3f} s**********'
                        logger.info(info.format(model_name, dataset_name, v_type, nc, whitening, time.time() - start_time))
        print('========================================')
        info = '**********Finishing test mdoel: {}. Time use: {:.3f} s**********'
        logger.info(info.format(model_name, time.time() - s))
        print('========================================')
        f = open(RESULT_SAVE_PATH, 'w', encoding='utf-8')
        json.dump(res, f, ensure_ascii=False, indent=4)
    logger.info('**********Finish STS Experiments. Time use: {:.3f} min**********'.format((time.time() - start) / 60))


if __name__ == '__main__':
    import fire
    fire.Fire()
