# Tuning LangChain ReAct Agents with Torch-Tuner CLI

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
programmer.

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

 - `\nNew Input:`
   - This is new input from the user.
 - `\nThought:`
   - Decision step
   - "Do I need to use a tool?"
   - ALWAYS follows `\nNew Input:` step
 - `\nAction:`
   - Tool to use 
   - Only happens if answer to decision step is "Yes"
   - ALWAYS follows `\nThought:` step
 - `\nAction Input:`
   - Tool input(tool function arguments)
   - ALWAYS follows `\nAction:` step
 - `\nObservation:`
   - An observation about the tool output
   - ALWAYS follows `\nAction Input:` step
   - **IMPORTANT NOTE** - The `\nObservation:` token is sent as a stop sequence to the LLM by the agent
 - `\nFinal Answer:`
   - The agents final response to the user
   - ALWAYS last
   - Follows `\nThought:` step when the answer to "Do I need to use a tool?" is "No."
   - Follows `\nObservation:` step when the answer to "Do I need to use a tool?" is "Yes."