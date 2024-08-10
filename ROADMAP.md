# Torch Tuner CLI Roadmap

This project started as a learning exercise, but has turned into an effort 
to simplify/modularize Supervised-Fine-Tuning(SFT) of LLM models
with LoRA.

## Main Goals

- Fine-Tune any LLM model with a single simple CLI command
- Serve LLM models over REST APIs in an efficient, predictable and production-ready way
- REST API for automated SFT
- Javascript UI for APIs(this is a BIG STRETCH goal)

### TODOs

I plan to add a public [Trello](https://trello.com/) board for this project at some point,
but in the meantime I will track work/needs/bugs/requests here.

- Add production wrapper to LLM REST server
- Add ability to provide special tokens
- Add support for non-llama models
  - Mistral
  - Falcon
  - Alpaca
  - BERT
  - ETC...
- Optimize quantization(QLoRA)
- Add `/sft/api/v1/tune` endpoint
- Add documentation comments to all argument classes to describe the individual arguments in more detail
- Reduce LLM server memory usage
- Add ability to request specific adapters from completions endpoints
  - Probably leveraging the model argument that is currently ignored
- Add CPU based SFT
- Add ability to configure training evaluations
- Add ability to prepare/configure/load more advanced datasets