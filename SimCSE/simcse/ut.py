# coding: utf-8
import os
import sys

from transformers import BertConfig, TrainingArguments, HfArgumentParser

from models import BertForCL
from args import ModelArguments, DataTrainingArguments

p = '/source/c0/NLPSource/embedding/transformer_based/chinese_simbert_L-12_H-768_A-12'
config = BertConfig.from_pretrained(p, hidden_dropout_prob=0.3)

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

model = BertForCL.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool('.ckpt' in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            model_args=model_args
        )
print(model)
