# coding: utf-8
import logging
from dataclasses import dataclass, field
from typing import Optional

from transformers import MODEL_FOR_MASKED_LM_MAPPING

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    '''
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    '''

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'The model checkpoint for weights initialization.'
            "Don't set if you want to train a model from scratch."
        },
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={
            'help': 'The model name.'
            'must set.'
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={'help': 'If training from scratch, pass a model type from the list: ' + ', '.join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={'help': 'Pretrained config name or path if not the same as model_name'}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'Where do you want to store the pretrained models downloaded from huggingface.co'},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={'help': 'Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.'},
    )
    model_revision: str = field(
        default='main',
        metadata={'help': 'The specific model version to use (can be a branch name, tag name or commit id).'},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            'help': 'Will use the token generated when running `transformers-cli login` (necessary to use this script '
            'with private models).'
        },
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            'help': 'Temperature for softmax.'
        }
    )
    pooler_type: str = field(
        default='cls',
        metadata={
            'help': 'What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last).'
        }
    )
    hard_negative_weight: float = field(
        default=0,
        metadata={
            'help': 'The **logit** of weight for hard negatives (only effective if hard negatives are used).'
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            'help': 'Whether to use MLM auxiliary objective.'
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            'help': 'Weight for MLM auxiliary objective (only effective if --do_mlm).'
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            'help': 'Use MLP only during training'
        }
    )
    hidden_dropout_prob: float = field(
                                    default=0.3,
                                    metadata={
                                        'help': 'Hidden dropout'
                                        }
                                    )
    dimension_reduction: bool = field(
                                    default=True,
                                    metadata={
                                        'help': 'Dimension reduction'
                                        }
                                    )
    output_size: int = field(
                            default=256,
                            metadata={'help': 'Dimension after dimension reduction'}
    )


@dataclass
class DataTrainingArguments:
    '''
    Arguments pertaining to what data we are going to input our model for training and eval.
    '''

    # Huggingface's original arguments.
    dataset_name: Optional[str] = field(
        default=None, metadata={'help': 'The name of the dataset to use (via the datasets library).'}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={'help': 'The configuration name of the dataset to use (via the datasets library).'}
    )
    overwrite_cache: bool = field(
        default=False, metadata={'help': 'Overwrite the cached training and evaluation sets'}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            'help': "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={'help': 'The number of processes to use for the preprocessing.'},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None,
        metadata={'help': 'The training data file (.txt or .csv).'}
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The training data cache dir'}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            'help': 'The maximum total input sequence length after tokenization. Sequences longer '
            'than this will be truncated.'
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            'help': 'Whether to pad all samples to `max_seq_length`. '
            'If False, will pad the samples dynamically when batching to the maximum length in the batch.'
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={'help': 'Ratio of tokens to mask for MLM (only effective if --do_mlm)'}
    )
    eval_result_save_path: str = field(
        default='',
        metadata={'help': 'Eval result save path)'}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None:
            raise ValueError('Need either a dataset name or a training file.')
        else:
            if self.train_file is not None:
                extension = self.train_file.split('.')[-1]
                assert extension in ['csv', 'json', 'txt'], '`train_file` should be a csv, a json or a txt file.'
