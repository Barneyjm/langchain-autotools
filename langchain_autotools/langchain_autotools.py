"""Generate LangChain tools and toolkits from any Python SDK.

``AutoToolWrapper`` introspects an arbitrary SDK client, decides which of its
methods an agent is allowed to reach via ``CrudControls``, and turns each of
them into an ``AutoTool`` -- a ``BaseTool`` with an ``args_schema`` derived from
the underlying function signature, so modern tool-calling models can fill in
arguments directly instead of hand-rolling a JSON blob.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Iterable, Iterator
from fnmatch import fnmatch
from json import JSONDecodeError
from re import Pattern
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import BaseModel, ConfigDict, PrivateAttr, create_model, model_validator

CRUD_TYPES = ("create", "read", "update", "delete")

# Default CRUD toggles: read-only unless the caller opts in.
AUTOTOOL_CRUD_CONTROLS_CREATE = False
AUTOTOOL_CRUD_CONTROLS_READ = True
AUTOTOOL_CRUD_CONTROLS_UPDATE = False
AUTOTOOL_CRUD_CONTROLS_DELETE = False

# Default patterns. Plain wildcards are glob patterns; anything using regex-only
# syntax (anchors, groups, quantifiers, ...) is compiled as a regex instead.
AUTOTOOL_CRUD_CONTROLS_CREATE_LIST = ["create_*"]
AUTOTOOL_CRUD_CONTROLS_READ_LIST = ["get_*"]
AUTOTOOL_CRUD_CONTROLS_UPDATE_LIST = ["update_*"]
AUTOTOOL_CRUD_CONTROLS_DELETE_LIST = ["delete_*"]

# Characters that only mean something in a regex -- glob patterns use
# ``*``, ``?`` and ``[...]``, which are deliberately absent from this set.
_REGEX_ONLY_CHARS = frozenset(r".^$+{}|\()")

# Parameters that are part of the calling convention rather than the SDK call.
_FILTERED_PARAMS = frozenset({"self", "cls", "run_manager", "callbacks"})


class CrudControls(BaseModel):
    """Controls which SDK functions are exposed, grouped by CRUD verb.

    Each verb has an on/off switch and a list of patterns. A function is exposed
    when its name matches a pattern belonging to an enabled verb. Patterns may be
    globs (``"get_thing*"``) or regexes (``r"^get_thing_\\w+$"``); the style is
    detected per pattern, so the two can be mixed freely in one list.
    """

    create: bool = AUTOTOOL_CRUD_CONTROLS_CREATE
    create_list: list[str] = AUTOTOOL_CRUD_CONTROLS_CREATE_LIST
    read: bool = AUTOTOOL_CRUD_CONTROLS_READ
    read_list: list[str] = AUTOTOOL_CRUD_CONTROLS_READ_LIST
    update: bool = AUTOTOOL_CRUD_CONTROLS_UPDATE
    update_list: list[str] = AUTOTOOL_CRUD_CONTROLS_UPDATE_LIST
    delete: bool = AUTOTOOL_CRUD_CONTROLS_DELETE
    delete_list: list[str] = AUTOTOOL_CRUD_CONTROLS_DELETE_LIST

    # Per-instance cache; never share compiled patterns between instances.
    _compiled_patterns: dict[str, dict[str, list[Pattern | str]]] = PrivateAttr(
        default_factory=dict
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_none_values(cls, values: Any) -> Any:
        """Treat an explicit ``None`` as "use the default" for every field."""
        if isinstance(values, dict):
            return {k: v for k, v in values.items() if v is not None}
        return values

    def _is_regex_pattern(self, pattern: str) -> bool:
        """Return ``True`` when ``pattern`` should be treated as a regex.

        A pattern is a regex if it is explicitly written as one (``r"..."``) or
        if it uses syntax that has no meaning in a glob. ``"get_*"`` therefore
        stays a glob, while ``r"^get_[^_]+$"`` is compiled as a regex.
        """
        if pattern.startswith(('r"', "r'")):
            return True
        return any(char in _REGEX_ONLY_CHARS for char in pattern)

    def compile_patterns(self) -> None:
        """Compile the pattern lists for every CRUD verb."""
        self._compiled_patterns.clear()

        for crud_type in CRUD_TYPES:
            regex_patterns: list[Pattern] = []
            glob_patterns: list[str] = []

            for pattern in getattr(self, f"{crud_type}_list", []) or []:
                if not isinstance(pattern, str):
                    continue
                if self._is_regex_pattern(pattern):
                    if pattern.startswith(('r"', "r'")):
                        pattern = pattern[2:-1]
                    try:
                        regex_patterns.append(re.compile(pattern))
                    except re.error:
                        # Not a valid regex after all -- fall back to glob.
                        glob_patterns.append(pattern)
                else:
                    glob_patterns.append(pattern)

            self._compiled_patterns[crud_type] = {
                "regex": regex_patterns,
                "glob": glob_patterns,
            }

    def matches_pattern(self, func_name: str, crud_type: str) -> bool:
        """Return ``True`` if ``func_name`` is allowed under ``crud_type``."""
        if not getattr(self, crud_type, False):
            return False

        if not self._compiled_patterns:
            self.compile_patterns()

        patterns = self._compiled_patterns.get(crud_type, {"regex": [], "glob": []})

        if any(pattern.match(func_name) for pattern in patterns["regex"]):
            return True
        return any(fnmatch(func_name, pattern) for pattern in patterns["glob"])

    def allows(self, func_name: str) -> bool:
        """Return ``True`` if ``func_name`` matches any enabled CRUD verb."""
        return any(self.matches_pattern(func_name, crud) for crud in CRUD_TYPES)


def _unwrap_client(client: Any) -> Any:
    """Accept either a bare SDK client or the legacy ``{"client": sdk}`` form."""
    if isinstance(client, dict) and set(client) == {"client"}:
        return client["client"]
    return client


def _describe(func: Any, name: str) -> str:
    """Build a tool description from the function's docstring or signature."""
    doc = inspect.getdoc(func)
    if doc and doc.strip():
        return doc.strip()
    try:
        return f"Call the `{name}` operation with signature {inspect.signature(func)}."
    except (TypeError, ValueError):
        return f"Call the `{name}` operation."


def _build_args_schema(func: Any, name: str) -> type[BaseModel] | None:
    """Derive a pydantic args schema from ``func``'s signature.

    Returns ``None`` when the signature cannot be introspected, in which case the
    tool falls back to accepting a free-form payload.
    """
    try:
        try:
            signature = inspect.signature(func, eval_str=True)
        except (NameError, TypeError):
            # Unresolvable string annotations -- keep them unevaluated.
            signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None

    fields: dict[str, tuple[Any, Any]] = {}
    accepts_var_kwargs = False

    for param_name, param in signature.parameters.items():
        if param_name in _FILTERED_PARAMS:
            continue
        if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            # ``**kwargs``/``*args`` mean the real argument list is unknown, so
            # let anything through rather than inventing fake parameters.
            accepts_var_kwargs = True
            continue

        annotation = param.annotation
        if annotation is inspect.Parameter.empty or isinstance(annotation, str):
            annotation = Any
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (annotation, default)

    if not fields and accepts_var_kwargs:
        # Nothing is known about the arguments (a dynamic ``(*args, **kwargs)``
        # client, say). An empty schema would swallow every argument, so leave
        # the tool schema-less and let the payload through untouched.
        return None

    safe_name = re.sub(r"\W|^(?=\d)", "_", name)
    model_name = f"{safe_name}Schema"
    config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow" if accepts_var_kwargs else "ignore",
    )
    try:
        return create_model(model_name, __config__=config, **fields)
    except Exception:  # noqa: BLE001 - exotic annotations shouldn't break tooling
        return None


class AutoTool(BaseTool):
    """A ``BaseTool`` bound to a single function on an SDK client."""

    client: Any
    name: str
    description: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_client(cls, values: Any) -> Any:
        if isinstance(values, dict) and "client" in values:
            values = {**values, "client": _unwrap_client(values["client"])}
        return values

    @classmethod
    def from_client(cls, client: Any, name: str, **kwargs: Any) -> AutoTool:
        """Build a tool for ``client.<name>``, inferring schema and description."""
        client = _unwrap_client(client)
        func = getattr(client, name)
        kwargs.setdefault("description", _describe(func, name))
        kwargs.setdefault("args_schema", _build_args_schema(func, name))
        return cls(client=client, name=name, **kwargs)

    @property
    def func(self) -> Any:
        """The bound SDK callable this tool wraps."""
        return getattr(self.client, self.name)

    def _coerce_input(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Support the legacy single-payload call style alongside kwargs.

        Modern tool calls arrive as keyword arguments validated against
        ``args_schema``. Older callers passed a JSON string or a dict as one
        positional argument, so those are still unpacked here.
        """
        if len(args) == 1 and isinstance(args[0], (str, dict)):
            payload = args[0]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except JSONDecodeError:
                    payload = {}
            if isinstance(payload, dict):
                return (), {**payload, **kwargs}
        return args, kwargs

    @staticmethod
    def _serialize(result: Any) -> str:
        if isinstance(result, (Iterator, Iterable)) and not isinstance(
            result, (str, bytes, dict)
        ):
            result = list(result)
        return json.dumps(result, default=str)

    def _run(
        self,
        *args: Any,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            func = self.func
        except AttributeError:
            return f"Invalid function name: {self.name}"

        args, kwargs = self._coerce_input(args, kwargs)
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            result = asyncio.run(_await(result))
        return self._serialize(result)

    async def _arun(
        self,
        *args: Any,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            func = self.func
        except AttributeError:
            return f"Invalid function name: {self.name}"

        args, kwargs = self._coerce_input(args, kwargs)
        if inspect.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            # Don't block the event loop on a synchronous SDK call.
            result = await asyncio.to_thread(func, *args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        return self._serialize(result)


async def _await(awaitable: Any) -> Any:
    return await awaitable


class AutoToolWrapper(BaseToolkit):
    """Turn an SDK client into a LangChain toolkit.

    ```python
    import boto3

    from langchain_autotools import AutoToolWrapper, CrudControls

    toolkit = AutoToolWrapper(
        client=boto3.client("s3"),
        crud_controls=CrudControls(read=True, read_list=["list_buckets"]),
    )
    tools = toolkit.get_tools()
    ```
    """

    client: Any
    operations: list[AutoTool] = []
    crud_controls: CrudControls = CrudControls()

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _normalize_client(cls, values: Any) -> Any:
        if isinstance(values, dict) and "client" in values:
            values = {**values, "client": _unwrap_client(values["client"])}
        return values

    def model_post_init(self, __context: Any) -> None:
        if not self.operations:
            self.operations = self._build_operations()

    def _build_operations(self) -> list[AutoTool]:
        operations: list[AutoTool] = []

        for func_name in dir(self.client):
            if func_name.startswith("_"):
                continue
            if not self.crud_controls.allows(func_name):
                continue
            try:
                if not callable(getattr(self.client, func_name)):
                    continue
            except Exception:  # noqa: BLE001 - properties may raise on access
                continue
            operations.append(AutoTool.from_client(self.client, func_name))

        return operations

    def get_tools(self) -> list[BaseTool]:
        """Get the tools in the toolkit."""
        return list(self.operations)
