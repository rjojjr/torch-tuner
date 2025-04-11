# Torch Tuner CHANGELOG

## 2.3.2
- refactor push cmd to not load model into memory

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