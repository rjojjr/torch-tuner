# Torch Tuner CHANGELOG

## 2.4.0
- upgrade all dependencies
- add unit/integration test framework
- add support for OpenAI chat JSONL dataset format
- fix serve endpoint token count and request parsing
- fix serve endpoint CUDA errors
- add `--accepted-api-key` argument for serve mode
- extend JSONL dataset format support to include LangGraph / OpenAI tool-call schema
- fix bash installer script to install CLI from a non-master branch

## 2.3.1
- disable gradient accumulation by default
- add ability to use flash attention
- fix fine-tune when `--do-eval` is false
- add `--push-adapter` argument
- save tuner config with models/adapters

## 2.3.0
- upgrade all dependencies
- improve memory efficiency
- fix QLORA
- update agent tokens
- update default arguments
- update argument descriptions