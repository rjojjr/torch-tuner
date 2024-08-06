# Torch Tuner FAQ

This is meant to be a general FAQ for the Torch Tuner CLI.

This document is a work in progress, so please be patient.

## FAQs

### Question

Why do I receive the following warning when tuning my model?

```
Setting `save_embedding_layers` to `True` as the embedding layer has been resized during finetuning.
```

#### Answer

This warning occurs when tuning your model with the `--is-chat-model` argument is set to true
while the `--save-embeddings` argument is set to false. Setting the chat model specific tokens 
requires that the embeddings layer is resized.

### Question

Why do I receive the error when resuming tuning of my model?

```
An unexpected Exception has been caught: loaded state dict contains a parameter group that doesn't match the size of optimizer's group
```

#### Answer

This error usually occurs when you resume a tuning job with different value 
for the `--save-embeddings` argument than tuning job was initially run with.