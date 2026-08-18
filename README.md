# langchain-autotools

Generate LangChain Tools and Toolkits from any Python SDK -- no extra code required.

Built for **LangChain 1.x** (`langchain-core >= 1.0`).

```bash
pip install langchain-autotools
```

## How AutoTools makes a dynamic toolkit

AutoTools wraps any SDK client that exposes CRUD-ish operations (get, update, create,
delete) and turns them into an LLM-enabled toolkit without a hand-written tool per call.

There are two core pieces:

1. **AutoToolWrapper** -- wraps the SDK you pass in and emits one tool per allowed method
2. **CrudControls** -- controls the access an agent has to Create, Read, Update and Delete
   verbs, with flexible pattern matching

Each generated tool carries an `args_schema` derived from the underlying function
signature, so tool-calling models fill in real, typed arguments:

```python
from langchain_core.utils.function_calling import convert_to_openai_tool

convert_to_openai_tool(toolkit.get_tools()[0])
# {'type': 'function',
#  'function': {'name': 'get_thing',
#               'description': 'Gets Thing',
#               'parameters': {'type': 'object',
#                              'properties': {'thing_id': {'type': 'integer'}},
#                              'required': ['thing_id']}}}
```

## Quick Start

```python
import boto3

from langchain_autotools import AutoToolWrapper, CrudControls

# Create your SDK client
s3 = boto3.client("s3")

# Configure access controls
crud_controls = CrudControls(
    read=True,
    read_list=["list_buckets"],       # exact match
    create=True,
    create_list=["create_bucket*"],   # wildcard match
)

# Create the toolkit
toolkit = AutoToolWrapper(client=s3, crud_controls=crud_controls)

# View available tools
print([tool.name for tool in toolkit.get_tools()])
```

`AutoToolWrapper` is a LangChain `BaseToolkit`, so `get_tools()` is the idiomatic way to
read its tools. The `toolkit.operations` attribute holds the same list.

## Pattern Matching in CrudControls

`CrudControls` supports both regex patterns and glob-style wildcards for matching
function names. The style is detected per pattern, so a single list can mix the two.

```python
# Glob patterns (simple wildcards)
crud_controls = CrudControls(
    read=True,
    read_list=["get_thing*"],  # Matches: get_thing, get_thing_by_id, get_thing_generator
)

# Regex patterns (more precise control)
crud_controls = CrudControls(
    read=True,
    read_list=[
        r"^get_thing$",      # Exact match only
        r"^get_thing_\w+$",  # Require at least one word char after the underscore
    ],
)

# Mix and match both styles
crud_controls = CrudControls(
    read=True,
    read_list=[
        "get_thing*",        # Glob pattern for broad matching
        r"^get_other_\w+$",  # Regex pattern for precise matching
    ],
)
```

Glob patterns support familiar shell-style wildcards:

- `*` matches everything
- `?` matches any single character
- `[seq]` matches any character in seq
- `[!seq]` matches any character not in seq

A pattern is treated as a regex when it uses syntax a glob has no meaning for -- anchors,
groups, quantifiers, escapes (`^ $ . + { } | \ ( )`) -- or when it is written explicitly
as `r"..."`. Everything else is a glob.

The defaults are read-only: `read=True` with `read_list=["get_*"]`, and create, update and
delete switched off.

### Excluding matches

`exclude` is a veto applied after the verb lists, so a wildcard can stay broad while
specific matches are dropped. It matters more than it sounds: `boto3` clients carry
`get_paginator` and `get_waiter`, which match `get_*`, become tools, and return an object
repr like `"<botocore.client.S3.Paginator.ListObjectsV2 object at 0x7f1a...>"` when an
agent calls them.

```python
crud_controls = CrudControls(
    read=True,
    read_list=["get_*", "list_*"],
    exclude=["get_paginator", "get_waiter", "*_internal"],
)
```

Exclude patterns use the same glob-or-regex detection as the verb lists.

## Working with Agents

`toolkit.get_tools()` returns a plain list of tools, so it drops straight into
`create_agent`:

```python
from langchain.agents import create_agent

agent = create_agent(
    "anthropic:claude-opus-5",
    toolkit.get_tools(),
    system_prompt="You are an assistant that manages AWS resources.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "How many S3 buckets do I have?"}]})
print(result["messages"][-1].content)
```

_NOTE:_ If your SDK has many callable functions, your tool list could exceed your model's
context length. Use `CrudControls` pattern matching to limit the tools your agent has
access to -- and see [Controlling description size](#controlling-description-size), which
is usually the larger cost of the two.

## Controlling description size

Descriptions come from the wrapped function's docstring, and some SDKs write essays.
Wrapping `boto3`'s S3 client with `read_list=["get_*", "list_*"]` produces 49 tools
carrying **~128,000 tokens** of description between them -- `get_object` alone is ~9,000.
The tool *count* is rarely the context problem; the docstrings are.

```python
# leading paragraph only
toolkit = AutoToolWrapper(client=sdk, describe="summary")

# hard ceiling, applied after `describe`
toolkit = AutoToolWrapper(client=sdk, max_description_length=1500)

# full control
toolkit = AutoToolWrapper(client=sdk, describe=lambda func, name: my_summary(func))
```

On the S3 example above, `describe="summary"` gives ~1,050 tokens -- a 99.2% cut -- with
`get_object` reading `Retrieves an object from Amazon S3.` A cap of 1500 gives ~17,800.

_Watch out:_ for a **dynamic** SDK (see [Dynamic SDKs](#dynamic-sdks)) no `args_schema`
could be derived, so the docstring is the only record of what a call accepts. Trimming it
takes that away -- with `describe="summary"`, nothing tells the model that `get_object`
needs a `Bucket` and a `Key`. Models often know popular SDKs well enough to manage, and
`fixed_args` can supply the rest, but test it before trusting it. SDKs with real
signatures are unaffected: their schema carries the arguments.

## Pinned arguments

`fixed_args` supplies values the model never chooses and never sees. They are stripped
from the `args_schema` and merged in at call time, overriding anything the model sent:

```python
toolkit = AutoToolWrapper(client=s3, fixed_args={"Bucket": "my-scoped-bucket"})
```

A pin is only applied to operations that can accept it -- one that takes no `Bucket` is
left alone rather than being called with an argument it would reject.

## Errors

An SDK exception is re-raised as a `ToolException` and, by default, handed back to the
agent as the tool's result (`"PermissionError: AccessDenied: not authorized"`) so it can
read the failure and adapt instead of the run ending. Set `handle_tool_error=False` on a
tool to let it propagate; the original exception is kept as `__cause__`.

## Raw results

Tools return a JSON string. To also get the underlying Python object without re-parsing,
ask for an artifact:

```python
toolkit = AutoToolWrapper(client=sdk, response_format="content_and_artifact")

message = tool.invoke(
    {"name": "get_thing", "args": {"thing_id": 3}, "id": "1", "type": "tool_call"}
)
message.content   # '{"status": 200, "response": {"id": 3}}'  -- what the model sees
message.artifact  # {'status': 200, 'response': {'id': 3}}    -- the real object
```

Generators are drained once, so the artifact is the materialized list.

## Capping result size

Descriptions are only half the context problem. A single realistic list call:

```python
list_objects(count=5000)  # -> 890,014 chars (~222,500 tokens)
```

That lands *mid-run*, after the agent has already committed to the call. Set a ceiling:

```python
toolkit = AutoToolWrapper(client=sdk, max_result_length=4000)
```

A list is cut on an item boundary, so the kept portion is still valid JSON. Anything else
is cut on a character boundary. Either way the result ends with a marker naming what was
dropped, so the model can tell "that is all of it" from "there is more":

```
[{"Key": "part-00000.parquet", ...}, ...]

[truncated: 4,992 of 5,000 items omitted. Narrow the request or raise max_result_length.]
```

Truncation applies to the content the model sees. If you also asked for an artifact, the
artifact keeps the whole result -- calling code has no context window to protect.

## Knowing which tools are destructive

The CRUD verb that exposed each operation is recorded on the tool, so you can act on it
without re-deriving the match:

```python
tool.metadata["crud"]          # "read" | "create" | "update" | "delete"
tool.metadata["sdk_function"]  # the underlying method name, ignoring any prefix
```

Which is what you want for putting a human in front of the dangerous half:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

destructive = [
    tool.name
    for tool in toolkit.get_tools()
    if tool.metadata["crud"] in ("create", "update", "delete")
]
agent = create_agent(
    "anthropic:claude-opus-5",
    toolkit.get_tools(),
    middleware=[HumanInTheLoopMiddleware(interrupt_on=dict.fromkeys(destructive, True))],
)
```

When a name matches under more than one verb, the first of create, read, update, delete
wins, so the reported verb is stable.

## Wrapping more than one SDK

Two SDKs that both expose `get_object` would hand the agent two identically named tools.
`prefix` keeps them apart; dispatch still uses the real method name:

```python
s3_tools = AutoToolWrapper(client=s3, prefix="s3_").get_tools()
gcs_tools = AutoToolWrapper(client=gcs, prefix="gcs_").get_tools()

agent = create_agent("anthropic:claude-opus-5", s3_tools + gcs_tools)
```

Prefixes are validated at construction, since a tool name providers reject is only
discovered at inference:

```python
AutoToolWrapper(client=s3, prefix="s3.")
# ValueError: prefix 's3.' contains '.', which is not allowed in a tool name.
#             Use letters, digits, underscores or hyphens -- for example 's3_'.
```

## Argument documentation

Where a docstring documents its parameters, that text is attached to the matching schema
field, so the model gets more than a type. Google style (an `Args:` block) and Sphinx
style (`:param name:`) are both understood:

```python
def get_thing(self, thing_id: int) -> dict:
    """Gets a thing.

    Args:
        thing_id: Identifier of the thing, as issued by the registry.
    """
```

```json
{"thing_id": {"type": "integer",
              "description": "Identifier of the thing, as issued by the registry."}}
```

This survives `describe="summary"` -- the description is trimmed to `Gets a thing.` while
the schema keeps the argument detail.

## Async

Every generated tool implements the async interface. Coroutine functions on the SDK are
awaited directly; synchronous ones are run in a worker thread so they don't block the
event loop.

```python
await toolkit.get_tools()[0].ainvoke({"thing_id": 123})
```

## Dynamic SDKs

Some SDKs (boto3 among them) build their methods dynamically, so the signature is just
`(*args, **kwargs)`. There is nothing to derive a schema from, so those tools skip the
`args_schema` and pass whatever the model supplies straight through to the SDK call.

## AWS Authentication Example

If using AWS services, ensure you have your credentials available:

```python
import getpass
import os

os.environ["AWS_ACCESS_KEY_ID"] = getpass.getpass(prompt="AWS Access Key ID: ")
os.environ["AWS_SECRET_ACCESS_KEY"] = getpass.getpass(prompt="AWS Secret Access Key: ")
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
```

## Development

```bash
poetry install
poetry run pytest
poetry run ruff check .
```
