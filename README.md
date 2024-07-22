# Torch Tuner README

This project serves as a simple convenient wrapper for fine-tuning 
llama based LLM models with simple text using LoRA and Torch.

Ideally, in the future, this project will support more complex training data structures
as well as fine-tuning vision and speech models.

## Running the Tuner

You should probably use a virtual environment
when installing dependencies and running the app, but that is up to you.

### Install Dependencies

From the project root, run:

```shell
pip install -r requirements.txt
```

### Run the Tuner

From the project root, run:

```shell
python src/tune/tune.py <your args>
```

To List Available Arguments:

```shell
python src/tune/tune.py -h
```

## CONTRIBUTING

Please feel free to submit a PR if you would to contribute to 
this project.