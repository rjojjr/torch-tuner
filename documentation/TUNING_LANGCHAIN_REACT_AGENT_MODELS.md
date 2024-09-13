# Tuning LangChain ReAct Agent Compatible Models with Torch-Tuner CLI

If you are looking to fine-tune LLM models for use with [LangChain](https://www.langchain.com/) 
[ReAct agents](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/),
don't worry, you're in the right place. I have put extensive work into the
Torch Tuner CLI to make tuning basic agent models easier. 

This document is meant to serve as a summary of everything you need to 
know inorder to fine-tune models for use with basic LangChain ReAct agents using the torch-tuner CLI.
As with anything else, your actual milage may vary.

## ReAct Agent Thought Process

The magic of LangChain ReAct agents lies in their ability to "think" 
through and solve complex multistep problems, as well as their ability
to chose from, and use, a wide variety of tools that are provided by their
programmers.

### Fork in the Road: Two Agent Paths

Each time a basic ReAct agent receives a request from a user,
the agent has to choose between one of two paths:

- To USE a Tool
- To NOT USE a Tool

The decision process always starts with the same thought: "Do I need to use a tool?".
If the answer is "No"(the reason for the use of double quotes will become apparent further down in this document),
the agent will proceed to form its "Final Answer" based on
the information it has at hand. If the answer is "Yes", the agent must figure out
which tool to use, and what input to provide the tool(if any).

#### Basic ReAct Agent Tokens

We now arrive at the next subject: Basic Agent Tokens. These "tokens" are used
by the basic ReAct agent to signify the different steps in its logical flow.

 - `\nNew Input`
   - This is new input from the user.
 - `\nThought`
   - Decision step
   - "Do I need to use a tool?"
   - ALWAYS follows `\nNew Input` step
 - `\nAction`
   - Tool to use 
   - Only happens if answer to decision step is "Yes"
   - ALWAYS follows `\nThought` step
 - `\nAction Input`
   - Tool input(tool function arguments)
   - ALWAYS follows `\nAction` step
 - `\nObservation`
   - An observation about the tool output
   - ALWAYS follows `\nAction Input` step
   - **IMPORTANT NOTE** - The `\nObservation` token is sent as a stop sequence to the LLM by the agent
 - `\nFinal Answer`
   - The agents final response to the user
   - ALWAYS last
   - Follows `\nThought` step when the answer to "Do I need to use a tool?" is "No."
   - Follows `\nObservation` step when the answer to "Do I need to use a tool?" is "Yes."

## Fine-Tuning ReAct Agent Models

Now that we covered the basic background context, let's talk about actually fine-tuning 
a basic LangChain ReAct Agent model. This document will assume that you are using 
the traditional Open AI JSONL tuning-data format(`{"prompt": "a prompt", "completion": "an expected AI response"}`),
though it should work in other formats as well.

### CLI Arguments

The first thing to do when tuning a new agent model, is set the `--use-agent-tokens` CLI argument to
'true'. That argument inserts the proper agent "tokens" from the related section above into
the model's vocabulary. This argument must be the same for both the "fine-tune" and "merge"
CLI steps.

You may also want to set the `--is-instruct-model` CLI argument to 'true', but that is optional.
That argument makes optimizations for tuning instructional flavored models, which may, or may not, help 
in your agent tuning endeavours.

### Tuning-Data Sample Formatting

As mentioned above, there are two scenarios/paths you can tune your 
agent model for: Use a Tool or NOT Use a Tool. 

#### No-Tool Sample

The 'No Tool' scenario is the most simple and intuitive case.

```JSONL
{"prompt": "\nNew Input: Hi!", "completion": "\nThought: Do I need to use a tool? No.\nFinal Answer: Hi. How are you?"}
```

#### Tool Sample

Since the agent sends the `\nObservation` "token" as a stop sequence, we must set
all completions to end with the `\nObservation` stop sequence to ensure the agent
stops and calls the tool properly. If you don't do this, the model will never 
call the tools properly, or worse, it will start predicting tool outputs.

```JSONL
{"prompt": "\nNew Input: Hi!", "completion": "\nThought: Do I need to use a tool? Yes.\nAction: your_tool\nAction Input: some,args\nObservation:"}
```

## Conclusion

By combining the CLI arguments & training sample format in the sections above,
you will be well on your way to tuning LangChain ReAct agent compatible LLM models
with the torch-tuner CLI in no time. I do want to call out that you 
want to be careful not to tune your model too heavily with static tools, 
you want your model to be able to choose from the tools that are presented to it, that
way you don't have to tune a new model everytime you add/edit/remove an
agent's available tools.