import os.path
from typing import Union

from datasets import load_dataset as load_data_set, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from datasets.arrow_dataset import Dataset as ArrowDataset

from arguments.arguments import TuneArguments
from exception.exceptions import ArgumentValidationException
def parse_jsonl(data):
    entries = []
    for line in data:
        entry = eval(line)
        if 'messages' in entry:
            messages = entry['messages']
            prompt_parts = [msg['content'] for msg in messages if msg['role'] == 'system' or msg['role'] == 'user']
            completion_parts = [msg['content'] for msg in messages if msg['role'] == 'assistant']
            prompt = '\n'.join(prompt_parts)
            completion = '\n'.join(completion_parts)
        elif 'prompt' in entry and 'completion' in entry:
            prompt = entry['prompt']
            completion = entry['completion']
        else:
            raise ValueError('Invalid JSONL format')
        entries.append({'prompt': prompt, 'completion': completion})
    return entries

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
            file_path = f"{arguments.training_data_dir}{seperator}{arguments.train_file}"
            data = parse_jsonl(open(file_path).readlines())
            train_set = Dataset.from_pandas(pd.DataFrame(data))
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
        eval_data = parse_jsonl(open(arguments.eval_dataset).readlines())
        eval_set = Dataset.from_pandas(pd.DataFrame(eval_data))
        train_set['eval'] = eval_set
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
