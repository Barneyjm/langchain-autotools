# langchain-autotools
Generate Tools and Toolkits from any Python SDK -- no extra code required

# How to use AutoTools to make a dynamic toolkit

AutoTools allows you to wrap any SDK client that exposes various CRUD operations (get, update, create, delete) and turn them into LLM-enabled toolkits without having to write a custom toolkit.

There are two core aspects to AutoTools:
1. AutoToolWrapper -- wraps the SDK you pass in
2. CrudControls -- allows you to control the access an agent has to Create, Read, Update, Delete verbs with flexible pattern matching

## Pattern Matching in CrudControls

CrudControls now supports both regex patterns and glob-style wildcards for matching function names:

```python
# Using glob patterns (simple wildcards)
crud_controls = CrudControls(
    read=True,
    read_list=["get_thing*"]  # Matches: get_thing, get_thing_by_id, get_thing_generator, etc.
)

# Using regex patterns (more precise control)
crud_controls = CrudControls(
    read=True,
    read_list=[
        r"^get_thing$",      # Exact match only
        r"^get_thing_\w+$"   # Match get_thing_something but require at least one word char after _
    ]
)

# Mix and match both styles
crud_controls = CrudControls(
    read=True,
    read_list=[
        "get_thing*",        # Glob pattern for broad matching
        r"^get_other_\w+$"   # Regex pattern for precise matching
    ]
)
```

Glob patterns support familiar shell-style wildcards:
- `*` matches everything
- `?` matches any single character
- `[seq]` matches any character in seq
- `[!seq]` matches any character not in seq

## Quick Start Example

```python
import boto3
from langchain_autotools import AutoToolWrapper, CrudControls

# Create your SDK client
s3 = boto3.client("s3")
client = {"client": s3}

# Configure access controls
crud_controls = CrudControls(
    read=True,
    read_list=["list_buckets"],  # Exact match
    create=True,
    create_list=["create_bucket*"]  # Wildcard match
)

# Create the toolkit
autotool = AutoToolWrapper(client=client, crud_controls=crud_controls)

# View available tools
print([tool.name for tool in autotool.operations])
```

## Working with Agents

Since `autotool.operations` is a list of tools, we can pass that into an Agent upon initialization:

```python
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_anthropic import ChatAnthropic

# Get the prompt to use
prompt = hub.pull("hwchase17/structured-chat-agent")

# Create the agent
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
agent = create_structured_chat_agent(llm, autotool.operations, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=autotool.operations,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
)

# Use the agent
agent_executor.invoke(
    {"input": "How many S3 buckets do I have? Pass an empty string as the 'action_input'."}
)
```

_NOTE:_ If your SDK has many callable functions, your tool list could exceed your model's context length. Use CrudControls pattern matching to limit the tools your Agent has access to.

## AWS Authentication Example

If using AWS services, ensure you have your credentials available:

```python
import os
import getpass

os.environ["AWS_ACCESS_KEY_ID"] = getpass.getpass(prompt="AWS Access Key ID: ")
os.environ["AWS_SECRET_ACCESS_KEY"] = getpass.getpass(prompt="AWS Secret Access Key: ")
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
```

## Cost Note
As of writing, using Anthropic's Claude3 Sonnet model in this notebook will cost less than USD 0.50 to complete.