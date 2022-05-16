# coding: utf-8
import logging
import os
import sys

import transformers
from transformers import (
    CONFIG_MAPPING,
    BertConfig,
    AutoModelForMaskedLM,
    BertTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
    Trainer
)
from transformers.trainer_utils import is_main_process

from .args import ModelArguments, DataTrainingArguments
from .dataset import OurDataCollatorWithPadding, DatasetLoader
from .models import BertForCL

logger = logging.getLogger(__name__)


def train(**kwargs):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif kwargs:
        model_args, data_args, training_args = parser.parse_dict(kwargs)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f'Output directory ({training_args.output_dir}) already exists and is not empty.'
            'Use --overwrite_output_dir to overcome.'
        )

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f' distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info('Training/evaluation parameters %s', training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        'cache_dir': model_args.cache_dir,
        'revision': model_args.model_revision,
        'use_auth_token': True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = BertConfig.from_pretrained(model_args.config_name, hidden_dropout_prob=model_args.hidden_dropout_prob, **config_kwargs)
    elif model_args.model_name_or_path:
        config = BertConfig.from_pretrained(model_args.model_name_or_path, hidden_dropout_prob=model_args.hidden_dropout_prob, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning('You are instantiating a new config instance from scratch.')
    tokenizer_kwargs = {
        'cache_dir': model_args.cache_dir,
        'use_fast': model_args.use_fast_tokenizer,
        'revision': model_args.model_revision,
        'use_auth_token': True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = BertTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            'You are instantiating a new tokenizer from scratch. This is not supported by this script.'
            'You can do it from another script, save it, and load it from here, using --tokenizer_name.'
        )
    if model_args.model_name_or_path:
        model = BertForCL.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool('.ckpt' in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            model_args=model_args
        )
    else:
        raise NotImplementedError
        logger.info('Training new model from scratch')
        model = AutoModelForMaskedLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer, do_mlm=model_args.do_mlm)
    dataset_loader = DatasetLoader(max_seq_length=data_args.max_seq_length, pad_to_max_length=data_args.pad_to_max_length, tokenizer=tokenizer)
    train_dataset = dataset_loader(data_args.train_file, data_args.data_cache_dir, data_args.preprocessing_num_workers, data_args.overwrite_cache)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    train_result = trainer.train()
    trainer.save_model()
    output_train_file = os.path.join(training_args.output_dir, 'train_results.txt')
    with open(output_train_file, 'w') as writer:
        logger.info('***** Train results *****')
        for key, value in sorted(train_result.metrics.items()):
            logger.info(f'  {key} = {value}')
            writer.write(f'{key} = {value}\n')
    trainer.state.save_to_json(os.path.join(training_args.output_dir, 'trainer_state.json'))


if __name__ == '__main__':
    train()
