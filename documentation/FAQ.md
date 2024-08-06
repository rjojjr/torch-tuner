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