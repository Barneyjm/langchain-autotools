# Changelog

## [0.1.0] - 2026-08-18

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
