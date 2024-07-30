# Torch Tuner README

This project serves as a simple convenient wrapper for fine-tuning 
Llama(and other model types in the future) based LLM models with simple text samples using LoRA and Torch.

Ideally, in the future, this project will support more complex training data structures,
non-llama LLM types and fine-tuning vision and speech models.

Use this project to fine-tune a suitable(Llama only ATM) base model that exists on Huggingface.

## Running the Tuner

This tuner will fine-tune, merge and push your new model to Huggingface depending on the arguments
you run it with.

### Tuning Data

The tuner will load data from a text(`.txt`) file. It expects each sample
to consume exactly one line.

### Install Dependencies

You should probably use a virtual environment
when installing dependencies and running the app,
but that is up to you.

From the project root, run:

```shell
pip install -r requirements.txt
```

### Run the Tuner

The tuner requires that you have the `HUGGING_FACE_TOKEN` environment
variable set to the Huggingface auth token that you would like to use.

From the project root(using the virtual environment(if any) that you used to install its dependencies), run:

```shell
python src/main/main.py <your args>
```

To List Available Arguments:

```shell
python src/main/main.py -h
```

or

```shell
python src/main/main.py --help
```

## CONTRIBUTING

Please feel free to submit a PR if you would to contribute to 
this project.

### Feature Requests

To request a feature or modification, please
submit a Github Issue.