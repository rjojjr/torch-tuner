from typing import Union

from datasets import load_dataset as load_data_set, DatasetDict, Dataset, IterableDatasetDict, IterableDataset

from arguments.arguments import TuneArguments

def load_dataset(arguments: TuneArguments) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    if arguments.hf_training_dataset_id is not None:
        return load_data_set(arguments.hf_training_dataset_id, split='train')
    elif arguments.train_file.endswith(".jsonl"):
        seperator = "/" if not arguments.training_data_dir.endswith("/") else ""
        return load_data_set("json", data_files={"train": f"{arguments.training_data_dir}{seperator}{arguments.train_file}"})
    else:
        return load_data_set(arguments.training_data_dir, data_files={"train": arguments.train_file})