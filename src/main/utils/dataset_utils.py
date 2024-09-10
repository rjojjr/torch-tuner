import os.path
from typing import Union

from datasets import load_dataset as load_data_set, DatasetDict, Dataset, IterableDatasetDict, IterableDataset

from arguments.arguments import TuneArguments


def load_dataset(arguments: TuneArguments) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """Load dataset for SFT trainer."""
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


def _load_eval_ds(arguments: TuneArguments, train_set: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    if arguments.eval_dataset is None:
        train_set['eval'] = train_set['train']
        return train_set
    elif os.path.isfile(arguments.eval_dataset) and arguments.eval_dataset.strip().endswith('jsonl'):
        eval_set = load_data_set("json", data_files={"eval": arguments.eval_dataset})
        train_set['eval'] = eval_set['eval']
        return train_set
    elif os.path.isfile(arguments.eval_dataset):
        eval_set = load_data_set(arguments.eval_dataset.replace(arguments.eval_dataset.split(os.sep)[len(arguments.eval_dataset.split(os.sep)) - 1], ''), data_files={"eval": arguments.eval_dataset.split(os.sep)[len(arguments.eval_dataset.split(os.sep)) - 1]})
        train_set['eval'] = eval_set['eval']
        return train_set
    else:
        eval_set = load_data_set(arguments.eval_dataset, split='eval')
        train_set['eval'] = eval_set['eval']
        return train_set