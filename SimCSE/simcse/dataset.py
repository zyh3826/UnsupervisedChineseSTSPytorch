# coding: utf-8
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Tuple

import torch
from datasets import load_dataset
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase


class DatasetLoader:
    def __init__(self, max_seq_length: int, pad_to_max_length: bool, tokenizer) -> None:
        self.max_seq_length = max_seq_length
        self.pad_to_max_length = pad_to_max_length
        self.tokenizer = tokenizer

    def __call__(self, data_path: str, cache_dir: str, preprocessing_num_workers: int, overwrite_cache: bool):
        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
        # (the dataset will be downloaded automatically from the datasets Hub
        #
        # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
        # behavior (see below)
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        data_files = {}
        data_files["train"] = data_path
        extension = data_path.split(".")[-1]
        if extension == "txt":
            extension = "text"
        if extension == "csv":
            datasets = load_dataset(extension, data_files=data_files, cache_dir=cache_dir, delimiter="\t" if "tsv" in data_path else ",")
        else:
            datasets = load_dataset(extension, data_files=data_files, cache_dir=cache_dir)

        # Prepare features
        column_names = datasets["train"].column_names
        self.sent2_cname = None
        if len(column_names) == 2:
            # Pair datasets
            self.sent0_cname = column_names[0]
            self.sent1_cname = column_names[1]
        elif len(column_names) == 3:
            # Pair datasets with hard negatives
            self.sent0_cname = column_names[0]
            self.sent1_cname = column_names[1]
            self.sent2_cname = column_names[2]
        elif len(column_names) == 1:
            # Unsupervised datasets
            self.sent0_cname = column_names[0]
            self.sent1_cname = column_names[0]
        else:
            raise NotImplementedError
        train_dataset = datasets["train"].map(
            self.prepare_features,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
        )
        return train_dataset

    def prepare_features(self, examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[self.sent0_cname])

        # Avoid "None" fields
        for idx in range(total):
            if examples[self.sent0_cname][idx] is None:
                examples[self.sent0_cname][idx] = " "
            if examples[self.sent1_cname][idx] is None:
                examples[self.sent1_cname][idx] = " "

        sentences = examples[self.sent0_cname] + examples[self.sent1_cname]

        # If hard negative exists
        if self.sent2_cname is not None:
            for idx in range(total):
                if examples[self.sent2_cname][idx] is None:
                    examples[self.sent2_cname][idx] = " "
            sentences += examples[self.sent2_cname]

        sent_features = self.tokenizer(
            sentences,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length" if self.pad_to_max_length else False,
        )

        features = {}
        if self.sent2_cname is not None:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]

        return features


# Data collator
@dataclass
class OurDataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    mlm: bool = True
    mlm_probability: float = 0.15
    do_mlm: bool = False

    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if self.do_mlm:
            batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

        batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
