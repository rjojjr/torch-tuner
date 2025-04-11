# Torch Tuner CLI Roadmap

This project started as a learning exercise, but has turned into an effort 
to simplify/modularize Supervised-Fine-Tuning(SFT) of LLM models
with LoRA.

## Main Goals

- Fine-Tune any LLM model with a single simple CLI command
- Serve LLM models over REST APIs in an efficient, predictable and production-ready way
- REST API for automated SFT
- Add support for audio models
- Add support for image generation models
- Add support for vision models
- Javascript UI for APIs(this is a BIG STRETCH goal)

### TODOs

I plan to add a public [Trello](https://trello.com/) board(or maybe a Github Kanban board) for this project at some point,
but in the meantime I will track work/needs/bugs/requests here.

- Add production wrapper to LLM REST server
- ~~Add ability to provide special/regular tokens to model vocabulary~~
- ~~Add Windows OS support~~
- ~~Add support for non-llama models~~
  - This is mostly satisfied with the addition of the 'GENERIC' LLM type
  - ~~Mistral~~
  - ~~Falcon~~
  - ~~Alpaca~~
  - ~~BERT~~
  - ~~ETC...~~
- ~~Optimize quantization(QLoRA)~~
- Add `/sft/api/v1/tune` endpoint
- Add documentation comments to all argument classes to describe the individual arguments in more detail
- Reduce LLM server memory usage
- Add ability to request specific adapters from completions endpoints
  - Probably leveraging the model argument that is currently ignored
- ~~Add CPU based SFT~~
- Add ability to configure/add advanced tuning evaluations
- Add ability to prepare/configure/load more advanced datasets
- Add ability to set max concurrent request for LLM serve mode
  - Add queue for waiting requests
- Add multi-gpu support
- Add support for ignored OpenAI request properties
- Add embeddings endpoint to serve mode
- Add ability to serve models on CPU
- Add ability to use JSON configs instead of argument config
- Add change serve model with API endpoint
- Fix eval_loss stat is not printed
- Add ability to use unsloth for memory/GPU constrained setups
- ~~Add ability to use flash_attention~~
- Add AMD GPU/CPU support
- ~~Add ability to disable gradient accumulation~~