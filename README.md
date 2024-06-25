# langchain-autotools
Generate Tools and Toolkits from any Python SDK -- no extra code required

# How to use AutoTools to make a dynamic toolkit

AutoTools allows you to wrap any SDK client that exposes various CRUD operations (get, update, create, delete) and turn them into LLM-enabled toolkits without having to write a custom toolkit.

There are two core aspects to AnySDK:
1. AutoToolWrapper -- wraps the SDK you pass in
2. CrudControls -- allows you to control the access an agent has to Create, Read, Update, Delete verbs.

As of writing, using Anthropic's Claude3 Sonnet model in this notebook will cost less than USD 0.50 to complete.
First, we'll create a client SDK. Let's use AWS's `boto3` library. Ensure you have your credentials available or set them here. 

```
import getpass
import os

os.environ["AWS_ACCESS_KEY_ID"] = getpass.getpass(prompt="AWS Access Key ID: ")
os.environ["AWS_SECRET_ACCESS_KEY"] = getpass.getpass(prompt="AWS Secret Access Key: ")
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
```

Let's validate that your S3 client is able to make an API call. This `list_buckets` command should match how many buckets are in your account.

```
import boto3

s3 = boto3.client("s3")
buckets = s3.list_buckets()
print(len(buckets["Buckets"]))
```

Now, we can pass this client into AutoToolWrapper to create the toolkit. The callable functions within the SDK will become available as tools within the `autotool.operations` object.

_NOTE_ If your SDK has many callable functions, your tool list could exceed your model's context length. Use CrudControls to limit the tools your Agent has access to.

To demonstrate this, we're limiting the `read_list` parameter to a single function, `list_buckets`. You can add more functions by using a prefix (ie, `list` or multiple values by using a comma separated list: `get,read,list`)

```
from langchain_autotools import AutoToolWrapper, CrudControls

client = {"client": s3}
crud_controls = CrudControls(
    read_list="list_buckets",
)

autotool = AutoToolWrapper(client=client, crud_controls=crud_controls)

print([tool.name for tool in autotool.operations])
```

Since `autotool.operations` is a list of tools, we can pass that into an Agent upon initialization and use them.

```
from langchain import hub

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/structured-chat-agent")
prompt.pretty_print()
# Now we'll set the Anthropic API key
os.environ["ANTHROPIC_API_KEY"] = getpass.getpass(prompt="Anthropic API Key: ")
With your API key set, we'll be able to create the AgentExecutor.
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

agent = create_structured_chat_agent(llm, autotool.operations, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=autotool.operations,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
)
```

Now we can use this agent to perform functions for us using the configured AWS `boto3` client! Let's ask the Agent how many buckets we have.

```
agent_executor.invoke(
    {
        "input": "How many S3 buckets do I have? Pass an empty string as the 'action_input'."
    }
)
```
Now, let's have our Agent create a bucket. We'll need to modify the CrudControls to include the `create_bucket` function from the `boto3` SDK.

We'll also need to enable the create actions as well by setting `create` to `True`.

_WARNING_ This will create a bucket in your account! (Buckets are free to create, so long as you don't store data in them.)

```
from langchain_autotools import AutoToolWrapper, CrudControls

client = {"client": s3}
crud_controls = CrudControls(
    read_list="list_buckets", create=True, create_list="create_bucket"
)

autotool = AutoToolWrapper(client=client, crud_controls=crud_controls)

print([tool.name for tool in autotool.operations])
```

More complicated tools often require more complex logic which requires more capable models. To create a bucket, we'll use the Sonnet model available from Anthropic.

```
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

agent = create_structured_chat_agent(llm, anysdk.operations, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=anysdk.operations,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
)
import random

rand = str(random.randint(0, 10000000))
agent_executor.invoke(
    {"input": f"Create a bucket named 'my-test-bucket-{rand}' in my account."}
)
```

To clean up the bucket you just created, run the below code block. Of course, you could also clean up with the Agent by modifying the CrudControls accordingly. 

```
s3.delete_bucket(Bucket=f"my-test-bucket-{rand}")
```