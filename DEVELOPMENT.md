# Torch Tuner DEVELOPMENT Notes

## Updating Dependencies

Use pip compile to update requirements.in(must have [CUDA Toolkit installed](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).):

```sh
pip-compile requirements.txt -o requirements.in --upgrade
```