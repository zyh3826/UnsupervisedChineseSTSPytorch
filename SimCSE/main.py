# coding: utf-8
import logging
import os
import time

from utils import (
                    MODEL_ROOT_PATH,
                    MODELS,
                    DATASET_ROOT_PATH,
                    DATASETS,
                    VEC_TYPE,
                    SAVE_ROOT)
from ToolFunction.SimCSE.train import train

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info('**********Begin STS Experiments**********')
    start = time.time()
    for model_name, model_path in MODELS.items():
        print('\n\n\n========================================')
        info = '**********Begining test model: {}**********'.format(model_name)
        logger.info(info)
        s = time.time()
        model_path = os.path.join(MODEL_ROOT_PATH, model_path)
        for dataset_name, length in DATASETS.items():
            dataset_path_root = os.path.join(DATASET_ROOT_PATH, dataset_name)
            for r in [True, False]:
                if not r:
                    dataset_path = os.path.join(dataset_path_root, 'extract_data.txt')
                    t = 'all_data'
                else:
                    dataset_path = os.path.join(dataset_path_root, 'random_10k.txt')
                    t = 'random_sample_10k_data'
                for v_type in VEC_TYPE:
                    for dimension_reduction in [True, False]:
                        kwargs = {
                                'model_name': model_name,
                                'model_name_or_path': model_path,
                                'train_file': dataset_path,
                                'num_train_epochs': 1,
                                'per_device_train_batch_size': 64 if 'large' not in model_name else 16,
                                'learning_rate': 1e-5,
                                'max_seq_length': length,
                                'eval_steps': 125000000,
                                "save_steps": 500000,
                                'pooler_type': v_type,
                                'mlp_only_train': False,
                                'overwrite_output_dir': True,
                                'temp': 0.05,
                                'do_train': True,
                                'do_eval': False,
                                'fp16': False,
                                'device': 'cuda:0',
                                'dimension_reduction': dimension_reduction,
                                'data_cache_dir': '/source/c0/NLPSource/Datasets/cache/'}
                        if dimension_reduction:
                            for output_size in [512, 256, 128, 64, 32]:
                                info = '**********Model: {}. Dataset: {}. Vector type: {} dimension_reduction: {}, output_size: {}.**********'
                                logger.info(info.format(model_name, dataset_name, v_type, dimension_reduction, output_size))
                                start_time = time.time()
                                tail = 'simcse-{}-{}-{}-{}-dimension_{}'.format(model_name, dataset_name, v_type, t, output_size)
                                if tail in os.listdir(os.path.join(SAVE_ROOT, model_name)):
                                    continue
                                output_dir = os.path.join(SAVE_ROOT, model_name, tail)
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                tmp = {
                                        'output_dir': output_dir,
                                        'output_size': output_size}
                                kwargs.update(tmp)
                                train(**kwargs)
                                info = '**********Model: {}. Dataset: {}. Vector type: {}, dimension_reduction: {}, output_size: {}.finishing. Time use: {:.3f} min**********'
                                logger.info(info.format(model_name, dataset_name, v_type, dimension_reduction, output_size, (time.time() - start_time) / 60))
                        else:
                            info = '**********Model: {}. Dataset: {}. Vector type: {}.**********'
                            logger.info(info.format(model_name, dataset_name, v_type))
                            start_time = time.time()
                            tail = 'simcse-{}-{}-{}-{}'.format(model_name, dataset_name, v_type, t)
                            if tail in os.listdir(os.path.join(SAVE_ROOT, model_name)):
                                continue
                            output_dir = os.path.join(SAVE_ROOT, model_name, tail)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            tmp = {'output_dir': output_dir}
                            kwargs.update(tmp)
                            train(**kwargs)
                            info = '**********Model: {}. Dataset: {}. Vector type: {}.finishing. Time use: {:.3f} min**********'
                            logger.info(info.format(model_name, dataset_name, v_type, (time.time() - start) / 60))
        print('========================================')
        info = '**********Finishing test model: {}. Time use: {:.3f} h**********'
        logger.info(info.format(model_name, (time.time() - s) / 3600))
        print('========================================')
    logger.info('**********Finish STS Experiments. Time use: {:.3f} h**********'.format((time.time() - start) / 3600))


if __name__ == '__main__':
    import fire
    fire.Fire(main)
