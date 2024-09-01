def add_agent_tokens(tokenizer, model):
    """Add LangChain ReAct Agent specific tokens to model vocabulary."""
    agent_tokens = ["\nThought:", "\nAction:", "\nAction Input:", "\nObservation:", "\nFinal Answer:", "\nNew Input:"]

    agent_tokens = set(agent_tokens) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(agent_tokens))
    if model is not None:
        model.resize_token_embeddings(len(tokenizer))


def add_additional_tokens(tokenizer, model, tokens):
    """Add specified tokens to model vocabulary."""
    vocab_tokens = set(tokens) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(vocab_tokens))
    if model is not None:
        model.resize_token_embeddings(len(tokenizer))