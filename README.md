# Torch Tuner CLI README

The torch-tuner project currently serves as a simple convenient CLI wrapper for supervised fine-tuning(and serving) 
LLM models(and others in the near future) on Nvidia CUDA enabled GPUs(or CPUs)
with simple text samples(or JSON Lines files) using [LoRA](https://github.com/microsoft/LoRA), [Transformers](https://huggingface.co/docs/transformers/en/index) and [Torch](https://en.wikipedia.org/wiki/Torch_(machine_learning)).

Use torch-tuner's CLI to perform Supervised Fine-Tuning(SFT)(with LoRA[and QLoRA]) of
a suitable base model that exists locally or on [Huggingface](https://huggingface.co) with simple text/JSONL.
You can also use this CLI to deploy your model(or any model)
as an REST API that mimics commonly used Open AI endpoints. You can
use this CLI to tune anything from basic chat and instructional models, to 
[LangChain ReAct Agent compatible models](documentation/TUNING_LANGCHAIN_REACT_AGENT_MODELS.md).

Ideally, in the future, the torch-tuner project will support more complex training data structures
and fine-tuning vision and speech models.

## Running the Torch Tuner CLI

The torch-tuner CLI will fine-tune, merge, push(to Huggingface) and/or serve your new fine-tuned model depending 
on the arguments you run it with.

### Using the Torch Tuner CLI

The torch-tuner CLI can be installed as a system-wide application, or run from source with python.
I typically wrap/configure my tuner CLI commands with bash scripts for convenience. You could also
use aliases to help keep your most commonly used CLI commands handy and easily accessible.
You might want to install the tuner CLI(using the instructions from the "Install Torch-Tuner CLI" section below) for 
easy access. 

I currently use this CLI across several different debian based OSes(across multiple machines), but it should
work on any OS. The torch-tuner CLI requires that you have the proper CUDA software/drivers(as well as python 3)
installed on the host. I would like to add CPU based tuning in the near future.

#### Install the Torch Tuner CLI

You can install the torch tuner CLI as a system-wide application on any OS(including Windows 
OS[although the linux script will probably work on WSL(Windows Subsystem for Linux), which you should probably be using anyway]) 
with [this script](scripts/install-torch-tuner.sh)(or [this script for Windows OS[NON-WSL]](scripts/win/install-torch-tuner.bat)) 
if you don't want to have to mess with python or the repository in general. After installation,
you can run the CLI with the `torch-tuner` command.

**NOTE** - You must run the script with root/admin privileges.

You can download the latest installer script from this repo
and execute it with one of the following single commands:

```shell
# Linux, MacOS & WSL
wget -O - https://raw.githubusercontent.com/rjojjr/torch-tuner/master/scripts/install-torch-tuner.sh | sudo bash

# Windows(non-WSL) (requires git & python3.11 already installed on target machine)
curl -sSL https://raw.githubusercontent.com/rjojjr/torch-tuner/master/scripts/win/install-torch-tuner.bat -o install-torch-tuner.bat && install-torch-tuner.bat && del install-torch-tuner.bat
```

**NOTE** - If the Unix installer script fails with OS level python dependency errors, and you are using Debian-Based Linux, 
try running the script with the `--install-apt-deps` flag. Otherwise, install the missing OS packages(python3, pip and python3-venv)
and run the torch-tuner CLI installer script again.

##### Updating Torch Tuner CLI

You can update the installed torch-tuner CLI instance at anytime by running the torch-tuner installer script again.

##### Uninstall Torch Tuner CLI

You can uninstall the torch-tuner CLI by running the uninstaller script:

```shell
# Linux, MacOS & WSL
sudo bash /usr/local/torch-tuner/scripts/uninstall-torch-tuner.sh

# Windows 
"%UserProfile%\AppData\Local\torch-tuner\scripts\win\uninstall-torch-tuner.bat"
```

#### Merging your LoRA Adapter

The results of your fine-tune job will be saved as a LoRA adapter. That LoRA adapter can then 
be merged with the base model to create a new model. Using the default arguments,
the tuner will merge your adapter and push the new model to a private Huggingface repository.

You can skip the fine-tune job by adding the `--fine-tune false` argument to your command.
You can also skip the merge or push by adding the `--push false` or `--merge false` arguments
to your command.

### Tuning Data

You can use a dataset from Huggingface by setting the `--hf-training-dataset-id` argument
to the desired HF repository identifier. Otherwise, you can use a local dataset by setting
both the `--training-data-dir` and `--training-data-file` CLI arguments.

#### Simple Text

The tuner will load plain-text data from a text-based file. It expects each training sample
to consume exactly one line. This might be useful for older models as well tuning a model with 
large amounts of plain/unstructured text.

#### JSON Lines(JSONL)

Torch Tuner accepts [JSONL](https://jsonlines.org/) training data in addition to plain text.

Accepted JSONL Formats:

```json lines
{"messages": [{"role": "system", "content": "You are helpful"}, {"role":  "user", "content":  "Hi!"}]}

OR

{"prompt": "<context & prompt>", "completion": "<ideal AI response>"}
```

### Install Dependencies

If you choose to not install the torch-tuner CLI, and run it from
the project root, you must install the CLI's 
python dependencies.

You should probably use a virtual environment
when installing dependencies and running the torch-tuner CLI,
but that is up to you.

From the torch-tuner project root, run:

```shell
pip install -r requirements.in
```

### Run the Torch Tuner CLI

The torch-tuner CLI currently requires that you have the `HUGGING_FACE_TOKEN` environment
variable set to a valid Huggingface authentication token in whatever shell you run it in.
This should most likely be added as an argument in the future(it is only really required for pushing/pulling models to/from HF, so this requirement should be removed).

From the torch-tuner CLI project root(using the virtual environment[if any] that you used to install torch-tuner's dependencies), run:

```shell
python3 src/main/main.py <your args>

# A Real Example
python3 src/main/main.py \
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
  --training-data-file samples.json \
  --epochs 10 \
  --lora-r 16 \
  --lora-alpha 32
```

To List Available Torch Tuner CLI Arguments:

```shell
python3 src/main/main.py --help
```

### Serve Mode(EXPERIMENTAL)

You can run the torch-tuner CLI in the new experimental serve mode to serve your model as a REST API that mimics the [Open AI](https://openai.com/)
completions(`/v1/completions` & `/v1/chat/completions`) endpoints.

```shell
python3 src/main/main.py \
  --serve true \
  --serve-model llama-tuned \
  --serve-port 8080
  
# When the Torch Tuner CLI is installed
torch-tuner \
  --serve true \
  --serve-model llama-tuned \
  --serve-port 8080
  
# Use dynamic quantization
python3 src/main/main.py \
  --serve true \
  --serve-model llama-tuned \
  --serve-port 8080 \
  --use-4bit true
```

The Open AI like REST endpoints will ignore the model provided in the request body, and will
always evaluate all requests against the model that is provided by the `--serve-model` argument.

**WARNING** - Serve mode is currently in an experimental state and should NEVER be used in a production environment.

### Useful Notes

Most of the default CLI arguments are configured to consume the least amount of memory possible.

You can find the supported arguments and their default values
[here](src/main/utils/argument_utils.py)(in the `_build_program_argument_parser` function)
if you don't want to run the CLI(`torch-tuner --help`) to find them.

You can easily extend this CLI to support more LLM types by following the pattern 
pointed out in the [Torch Tuner FAQ document](documentation/FAQ.md).

## Feature Requests

To request a feature or modification(or report a bug), please
submit a Github Issue. I gladly welcome and encourage any and all feedback.

### Roadmap

To view current plans for future work and futures, please take a look at
the [roadmap](ROADMAP.md)

## LICENSE

This project is [licensed](LICENSE.txt) under MIT. 