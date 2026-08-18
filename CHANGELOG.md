# Changelog

## [0.2.0] - 2026-08-18

### Added
- **Result truncation.** `max_result_length` caps what a call returns before it reaches
  the model -- lists cut on an item boundary so the kept portion stays valid JSON,
  everything else on a character boundary, both ending in a marker naming what was
  dropped. Artifacts keep the whole result. One realistic list call measured ~222k
  tokens, and it lands mid-run where the agent cannot recover.
- **Exclude patterns.** `CrudControls(exclude=[...])` vetoes matches after the verb
  lists, so wildcards can stay broad. `boto3` clients need it: `get_paginator` and
  `get_waiter` match `get_*` and return object reprs when called.
- **CRUD metadata.** Tools carry `metadata["crud"]` (the verb that exposed them) and
  `metadata["sdk_function"]`, so destructive operations can be routed to
  `HumanInTheLoopMiddleware` or an approval gate without re-deriving the match.
- **Name prefixes.** `prefix` renames tools so two wrapped SDKs don't both expose
  `get_object`; dispatch still uses the underlying method name. Validated at
  construction against the characters providers allow in a tool name, so `prefix="s3."`
  fails immediately rather than producing tools a model cannot call.
- **Argument documentation.** Per-parameter docstring text (Google `Args:` blocks and
  Sphinx `:param name:`) becomes the schema field's description, and survives
  `describe="summary"`.
- **Description budget.** `describe="summary"` keeps only a docstring's leading
  paragraph and `max_description_length` caps it outright, for SDKs whose docstrings
  dominate the prompt (49 S3 tools carry ~128k tokens of description by default).
  `describe` also accepts a `(func, name)` callable. Note that trimming a dynamic SDK's
  docstring removes the only record of what its calls accept, since no schema could be
  derived for it.
- **Pinned arguments.** `fixed_args` supplies values the model never sees or chooses.
  They are removed from the `args_schema`, merged in at call time, and only applied to
  operations that can accept them.
- **Errors reach the agent.** SDK exceptions are re-raised as `ToolException` and
  returned as the tool result by default, so an agent can read the failure and adapt
  rather than the run ending. `handle_tool_error=False` restores propagation; the
  original exception is preserved as `__cause__`.
- **Raw results.** `response_format="content_and_artifact"` returns the materialized
  Python object alongside the JSON string.

### Changed
- **LangChain 1.x.** The package now targets `langchain-core >= 1.0` and depends on
  `langchain-core` directly rather than the full `langchain` meta-package.
- **Tools carry a real `args_schema`.** Each generated tool derives a pydantic schema from
  the wrapped function's signature, so tool-calling models fill in typed arguments instead
  of hand-assembling a JSON string. Functions with a dynamic `(*args, **kwargs)` signature
  (boto3 and friends) keep the schema-less passthrough behaviour.
- **`AutoToolWrapper` is a `BaseToolkit`.** `get_tools()` is the idiomatic accessor;
  `operations` still holds the same list.
- **Clients can be passed directly.** `AutoToolWrapper(client=s3)` works; the older
  `AutoToolWrapper(client={"client": s3})` form is still accepted.
- **Real async.** `_arun` awaits coroutine SDK functions and runs synchronous ones in a
  worker thread instead of blocking the event loop.
- Default CRUD pattern lists are a single glob per verb (`["get_*"]`, `["create_*"]`, ...)
  rather than a regex/glob pair that overlapped.
- Packaging moved to PEP 621 metadata; Python 3.10+ is required.
- Ship a `py.typed` marker so downstream type checkers see the annotations.

### Fixed
- Glob patterns are no longer misdetected as regexes. `"get_thing*"` is now matched with
  `fnmatch` as documented, instead of being compiled to the regex `get_thing*` (which
  matched `get_thin` followed by any number of `g`).
- `CrudControls` instances no longer share one compiled-pattern cache through a mutable
  class attribute.
- Tools for functions without a docstring get a generated description instead of `None`,
  which newer LangChain versions reject.
- Generators are drained once, and the redundant dict branch in result serialization is
  gone.
- An `AttributeError` raised *inside* an SDK call is no longer reported as
  `Invalid function name`; only a genuinely missing method produces that message.
- A synchronous `invoke` on an async SDK method no longer raises
  `asyncio.run() cannot be called from a running event loop` when called from
  inside one.

### Removed
- **Breaking:** tools no longer accept a JSON string as their whole payload through the
  public `invoke`/`run` interface -- pass a dict of arguments instead
  (`tool.invoke({"thing_id": 123})`). Direct `_run` calls still unpack a single JSON
  string or dict payload.

## [0.0.2] - Initial Release

- Basic CRUD operations support
- Simple string matching for function names
- AWS SDK integration example
- Initial LangChain integration
