# Torch Tuner README

This project currently serves as a simple convenient CLI wrapper for fine-tuning(and serving) 
Llama(and others in the future) based LLM models on Nvidia GPUs with simple text samples using LoRA and Torch.

Use this project's CLI to fine-tune(with LoRA) a suitable(Llama only ATM) base model that exists on Huggingface with simple text and CUDA.
You can also use this model to deploy your model(or any model you have access to on Huggingface)
as an API that mimics commonly used Open AI endpoints.

Ideally, in the future, this project will support more complex training data structures,
non-llama LLM types and fine-tuning vision and speech models. Also, I would like this project
to be able to run as an API along with its current CLI implementation.

## Server Mode(EXPERIMENTAL)

You can run the CLI in the new experimental serve mode to serve your model over an API that mimics the Open AI 
completions(`/v1/completions` & `/v1/chat/completions`) endpoints.

```shell
python src/main/main.py \
  --serve true \
  --serve-model llama-tuned \
  --serve-port 8080
```

The endpoints will ignore the model provided in the request body, and will
always evaluate all requests on the model that is provided by the --serve-model argument.

## Running the Tuner

The tuner CLI will fine-tune, merge and push your new model to Huggingface depending on the arguments
you run it with.

### Using the Tuner

I typically wrap/configure my tuner CLI commands with bash scripts for convenience.
You might want to install the tuner CLI(using the instructions from the "Install Torch-Tuner CLI" section below) for 
easy access. 

I currently use this CLI across several different debian based OSes, but it should
work on any OS. The tuner requires that you have the proper CUDA software/drivers
installed on the host. I would like to add CPU based tuning in the future.

#### Install Torch-Tuner CLI

You can install the torch tuner as a system-wide command on any Linux OS(Windows support coming soon) 
with [this script](scripts/install-torch-tuner.sh). After installation,
you can run the CLI with the `torch-tuner` command.

**NOTE** - You must run the script with the `sudo` command.

You can download the script from [Github](https://raw.githubusercontent.com/rjojjr/torch-tuner/master/scripts/install-torch-tuner.sh)
and execute it with the following single command:

```shell
wget -O - https://raw.githubusercontent.com/rjojjr/torch-tuner/master/scripts/install-torch-tuner.sh | sudo bash
```

**NOTE** - If the installer script fails with dependency errors, and you are using Debian Linux, 
try running the script with the `--install-apt-deps` flag. Otherwise, install the missing OS packages(python3, pip and python3-venv) and run the installer again.

##### Uninstall

You can uninstall the torch-tuner CLI by running the uninstaller script:

```shell
sudo bash /usr/local/torch-tuner/scripts/uninstall-torch-tuner.sh
```

#### Merging your LoRA Adapter

The results of your fine-tune job will be saved as a LoRA adapter. That LoRA adapter can then 
be merged with the base model to create a new model. Using the default arguments,
the tuner will merge your adapter and push the new model to a private Huggingface repository.

You can skip the fine-tune job by adding the `--fine-tune false` argument to your command.
You can also skip the merge or push by adding the `--push false` or `--merge false` arguments
to your command.

### Tuning Data

The tuner will load data from a text-based file. It expects each training sample
to consume exactly one line. Currently, this app requires both the path to the directory
where the training data is stored and the training-data file name supplied as separate 
arguments.

### Install Dependencies

If you choose to not install the CLI, and run it from
the project root, you must install the torch-tuner's 
python dependencies.

You should probably use a virtual environment
when installing dependencies and running the CLI,
but that is up to you.

From the project root, run:

```shell
pip install -r requirements.txt
```

### Run the Tuner CLI

The tuner CLI currently requires that you have the `HUGGING_FACE_TOKEN` environment
variable set to a valid Huggingface authentication token in whatever shell you run it in.

From the project root(using the virtual environment(if any) that you used to install its dependencies), run:

```shell
python src/main/main.py <your args>

# A Real Example
python src/main/main.py \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --new-model llama-tuned \
  --training-data-dir /path/to/data \
  --training-data-file samples.txt \
  --epochs 10 \
  --lora-r 16 \
  --lora-alpha 32
  
# A Real Example with CLI Installed
torch-tuner \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --new-model llama-tuned \
  --training-data-dir /path/to/data \
  --training-data-file samples.txt \
  --epochs 10 \
  --lora-r 16 \
  --lora-alpha 32
```

To List Available Arguments:

```shell
python src/main/main.py --help
```

### Useful Notes

In theory, the base-model(`--base-model`) CLI argument will 
accept a path to a locally saved model instead of a Huggingface repository
name, but this is untested ATM.

You can find the supported arguments and their default values
[here](src/main/utils/argument_utils.py)(in the `_build_program_argument_parser` function)
if you don't want to run the CLI to find them.

## CONTRIBUTING

Please feel free to submit a PR if you would to contribute to 
this project.

### Feature Requests

To request a feature or modification, please
submit a Github Issue.

## LICENSE

This project is [licensed](LICENSE.txt) under MIT. 