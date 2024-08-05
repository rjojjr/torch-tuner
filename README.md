# Torch Tuner README

This project currently serves as a simple convenient CLI wrapper for fine-tuning(and serving) 
Llama based LLM models(and others in the near future) on Nvidia GPUs with simple text samples using LoRA and Torch.

Use this torch-tuner's CLI to fine-tune(with LoRA) a suitable(Llama only ATM) base model that exists locally or on Huggingface with simple text and CUDA.
You can also use this model to deploy your model(or any model you have access to)
as an REST API that mimics commonly used Open AI endpoints.

Ideally, in the future, this project will support more complex training data structures,
non-llama LLM types, CPU based tuning and fine-tuning vision and speech models. 

## Serve Mode(EXPERIMENTAL)

You can run the torch-tuner CLI in the new experimental serve mode to serve your model as a REST API that mimics the Open AI 
completions(`/v1/completions` & `/v1/chat/completions`) endpoints.

```shell
python src/main/main.py \
  --serve true \
  --serve-model llama-tuned \
  --serve-port 8080
  
# When the Torch Tuner CLI is installed
torch-tuner \
  --serve true \
  --serve-model llama-tuned \
  --serve-port 8080
  
# Use dynamic quantization
python src/main/main.py \
  --serve true \
  --serve-model llama-tuned \
  --serve-port 8080 \
  --use-4bit true
```

The Open AI like REST endpoints will ignore the model provided in the request body, and will
always evaluate all requests against the model that is provided by the `--serve-model` argument.

**WARNING** - Serve mode is currently in an experimental state and should NEVER be used in a production environment.

## Running Torch Tuner

The tuner CLI will fine-tune, merge, push(to Huggingface) and/or serve your new model depending on the arguments
you run it with.

### Using Torch Tuner

I typically wrap/configure my tuner CLI commands with bash scripts for convenience.
You might want to install the tuner CLI(using the instructions from the "Install Torch-Tuner CLI" section below) for 
easy access. 

I currently use this CLI across several different debian based OSes(across multiple machines), but it should
work on any OS. The torch-tuner CLI requires that you have the proper CUDA software/drivers(as well as python 3)
installed on the host. I would like to add CPU based tuning in the near future.

#### Install Torch-Tuner CLI

You can install the torch tuner CLI as a system-wide application on any Linux OS(Windows support coming soon[although this will probably work on WSL(Windows Subsystem for Linux), which you should probably be using anyways]) 
with [this script](scripts/install-torch-tuner.sh) if you don't want to have to mess with python or the repository in general. After installation,
you can run the CLI with the `torch-tuner` command.

**NOTE** - You must run the script with the `sudo` command.

You can download the latest installer script from [Github](https://raw.githubusercontent.com/rjojjr/torch-tuner/master/scripts/install-torch-tuner.sh)
and execute it with the following single command:

```shell
wget -O - https://raw.githubusercontent.com/rjojjr/torch-tuner/master/scripts/install-torch-tuner.sh | sudo bash
```

**NOTE** - If the installer script fails with OS level python dependency errors, and you are using Debian Linux, 
try running the script with the `--install-apt-deps` flag. Otherwise, install the missing OS packages(python3, pip and python3-venv)
and run the torch-tuner CLI installer script again.

##### Updating Torch Tuner CLI

You can update the installed torch-tuner CLI instance at anytime by running the torch-tuner installer script again.

##### Uninstall Torch Tuner CLI

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

If you choose to not install the torch-tuner CLI, and run it from
the project root, you must install the CLI's 
python dependencies.

You should probably use a virtual environment
when installing dependencies and running the torch-tuner CLI,
but that is up to you.

From the torch-tuner project root, run:

```shell
pip install -r requirements.txt
```

### Run the Torch Tuner CLI

The torch-tuner CLI currently requires that you have the `HUGGING_FACE_TOKEN` environment
variable set to a valid Huggingface authentication token in whatever shell you run it in.
I might add this as an argument in the future.

From the torch-tuner CLI project root(using the virtual environment(if any) that you used to install its dependencies), run:

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

To List Available Torch Tuner CLI Arguments:

```shell
python src/main/main.py --help
```

### Useful Notes

In theory, the base-model(`--base-model`) torch-tuner CLI argument will 
accept a path to a locally saved model instead of a Huggingface repository
name, but this is untested ATM.

You can find the supported arguments and their default values
[here](src/main/utils/argument_utils.py)(in the `_build_program_argument_parser` function)
if you don't want to run the CLI to find them.

## Feature Requests

To request a feature or modification(or report a bug), please
submit a Github Issue. I gladly welcome and encourage any and all feedback.

## LICENSE

This project is [licensed](LICENSE.txt) under MIT. 