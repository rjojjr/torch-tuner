# Torch Tuner CHANGELOG

## 2.4.1
- update installer script to support rootless install
- add support for LLM tuning on Apple Silicon Macs
- add support for Apple Silicon Macs to installer script
- add `uninstall` CLI command
- add `--use-gradient-checkpointing` CLI argument

## 2.4.0
- upgrade all dependencies
- add unit/integration test framework
- add support for OpenAI chat JSONL dataset format
- fix serve endpoint token count and request parsing
- fix serve endpoint CUDA errors
- add `--accepted-api-key` argument for serve mode
- extend JSONL dataset format support to include LangGraph / OpenAI tool-call schema
- fix bash installer script to install CLI from a non-master branch
- add case-insensitive file extension support
- fix broken precision/quantization options
- replace deprecated transformers `warmup_ratio` argument with `warmup_steps`
- fix `--fp32-cpu-offload` when using quantization

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