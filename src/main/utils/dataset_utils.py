import os.path
from typing import Union

from datasets import load_dataset as load_data_set, DatasetDict, Dataset, IterableDatasetDict, IterableDataset

from arguments.arguments import TuneArguments
from exception.exceptions import ArgumentValidationException


def load_dataset(arguments: TuneArguments) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """Load dataset for SFT trainer."""

    if arguments.do_train:
        print()
        print('Loading training dataset')
        print()
        if arguments.hf_training_dataset_id is not None:
            train_set = load_data_set(arguments.hf_training_dataset_id, split='train')
            if arguments.do_eval:
                train_set = _load_eval_ds(arguments, train_set)
            return train_set

        elif arguments.train_file.endswith(".jsonl"):
            seperator = os.sep if not arguments.training_data_dir.endswith(os.sep) else ""
            train_set = load_data_set("json", data_files={"train": f"{arguments.training_data_dir}{seperator}{arguments.train_file}"})
            if arguments.do_eval:
                train_set = _load_eval_ds(arguments, train_set)
            return train_set
        else:
            train_set = load_data_set(arguments.training_data_dir, data_files={"train": arguments.train_file})
            if arguments.do_eval:
                train_set = _load_eval_ds(arguments, train_set)
            return train_set
    else:
        return _load_eval_ds(arguments, DatasetDict({}))


def _load_eval_ds(arguments: TuneArguments, train_set: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """Load evaluation dataset."""

    print()
    print('Loading evaluation dataset')
    print()
    if 'train' in train_set and arguments.eval_dataset is None:
        train_set['eval'] = train_set['train']
        print()
        print('WARNING: You are using the training dataset as the evaluation dataset')
        print('If this is unintentional, please set the `--eval-dataset` CLI argument to your desired eval dataset.')
        print()
        return train_set
    elif os.path.isfile(arguments.eval_dataset) and arguments.eval_dataset.strip().endswith('jsonl'):
        eval_set = load_data_set("json", data_files={"eval": arguments.eval_dataset})
        train_set['eval'] = eval_set['eval']
        return train_set
    elif os.path.isfile(arguments.eval_dataset):
        eval_set = load_data_set(arguments.eval_dataset.replace(arguments.eval_dataset.split(os.sep)[len(arguments.eval_dataset.split(os.sep)) - 1], ''), data_files={"eval": arguments.eval_dataset.split(os.sep)[len(arguments.eval_dataset.split(os.sep)) - 1]})
        train_set['eval'] = eval_set['eval']
        return train_set
    elif (not 'train' in train_set) and arguments.eval_dataset is None:
        raise ArgumentValidationException('`--eval-dataset` argument is required for evaluation mode')
    else:
        eval_set = load_data_set(arguments.eval_dataset, split='eval')
        train_set['eval'] = eval_set['eval']
        return train_set