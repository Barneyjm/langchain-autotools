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
import logging
import re
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from json import JSONDecodeError
from re import Pattern
from typing import Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, BaseToolkit, ToolException
from pydantic import BaseModel, ConfigDict, PrivateAttr, create_model, model_validator

logger = logging.getLogger(__name__)

CRUD_TYPES = ("create", "read", "update", "delete")

#: How a tool description is built from the wrapped function.
DescriptionStyle = Literal["full", "summary"]

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


def _first_paragraph(text: str) -> str:
    """Return the leading paragraph of a docstring, collapsed onto one line."""
    lines: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            if lines:
                break
            continue
        lines.append(line.strip())
    return " ".join(lines)


def _describe(
    func: Any,
    name: str,
    *,
    style: DescriptionStyle | Callable[[Any, str], str] = "full",
    max_length: int | None = None,
) -> str:
    """Build a tool description from the wrapped function.

    ``style`` is ``"full"`` (the whole docstring), ``"summary"`` (its leading
    paragraph), or a callable taking ``(func, name)``. Note that for a dynamic
    SDK -- one whose signatures are ``(*args, **kwargs)``, so no ``args_schema``
    could be derived -- the docstring is the only record of what a call accepts,
    and shortening it takes that reference away from the model.
    """
    if callable(style):
        description = style(func, name)
    else:
        doc = inspect.getdoc(func)
        if doc and doc.strip():
            description = doc.strip()
            if style == "summary":
                description = _first_paragraph(description) or description
        else:
            try:
                signature = inspect.signature(func)
                description = f"Call the `{name}` operation with signature {signature}."
            except (TypeError, ValueError):
                description = f"Call the `{name}` operation."

    if max_length is not None and len(description) > max_length:
        description = description[:max_length].rstrip() + "..."
    return description


def _build_args_schema(
    func: Any, name: str, exclude: Iterable[str] = ()
) -> type[BaseModel] | None:
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
    excluded = set(exclude)

    for param_name, param in signature.parameters.items():
        if param_name in _FILTERED_PARAMS or param_name in excluded:
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


def _applicable_fixed_args(func: Any, fixed_args: dict[str, Any]) -> dict[str, Any]:
    """Narrow ``fixed_args`` to the ones ``func`` can actually receive.

    A toolkit-wide pin such as ``{"Bucket": "..."}`` is meaningful only for the
    operations that take a ``Bucket``; passing it to the others would raise a
    ``TypeError``. Functions with a ``**kwargs`` catch-all accept everything.
    """
    if not fixed_args:
        return {}
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return dict(fixed_args)

    names = set()
    for param_name, param in signature.parameters.items():
        if param.kind is param.VAR_KEYWORD:
            return dict(fixed_args)
        if param.kind is not param.VAR_POSITIONAL:
            names.add(param_name)
    return {k: v for k, v in fixed_args.items() if k in names}


class AutoTool(BaseTool):
    """A ``BaseTool`` bound to a single function on an SDK client."""

    client: Any
    name: str
    description: str
    fixed_args: dict[str, Any] = {}
    """Arguments pinned by the caller: merged into every call, hidden from the model."""

    # Surface SDK failures to the agent as a tool result rather than ending the
    # run. Set to False to let the ToolException propagate instead.
    handle_tool_error: bool | str | Callable[[ToolException], Any] | None = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_client(cls, values: Any) -> Any:
        if isinstance(values, dict) and "client" in values:
            values = {**values, "client": _unwrap_client(values["client"])}
        return values

    @classmethod
    def from_client(
        cls,
        client: Any,
        name: str,
        *,
        fixed_args: dict[str, Any] | None = None,
        describe: DescriptionStyle | Callable[[Any, str], str] = "full",
        max_description_length: int | None = None,
        **kwargs: Any,
    ) -> AutoTool:
        """Build a tool for ``client.<name>``, inferring schema and description."""
        client = _unwrap_client(client)
        func = getattr(client, name)
        pinned = _applicable_fixed_args(func, fixed_args or {})
        args_schema = _build_args_schema(func, name, exclude=pinned)
        kwargs.setdefault("args_schema", args_schema)
        kwargs.setdefault(
            "description",
            _describe(func, name, style=describe, max_length=max_description_length),
        )
        return cls(client=client, name=name, fixed_args=pinned, **kwargs)

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
    def _materialize(result: Any) -> Any:
        """Drain one-shot results (generators) so they can be read more than once."""
        if isinstance(result, (Iterator, Iterable)) and not isinstance(
            result, (str, bytes, dict)
        ):
            return list(result)
        return result

    def _format(self, result: Any) -> str | tuple[str, Any]:
        """Render the SDK result per ``response_format``."""
        result = self._materialize(result)
        content = json.dumps(result, default=str)
        if self.response_format == "content_and_artifact":
            return content, result
        return content

    def _prepare_call(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        args, kwargs = self._coerce_input(args, kwargs)
        # Pinned arguments are the caller's, not the model's -- they win.
        return args, {**kwargs, **self.fixed_args}

    def _run(
        self,
        *args: Any,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str | tuple[str, Any]:
        try:
            func = self.func
        except AttributeError:
            return f"Invalid function name: {self.name}"

        args, kwargs = self._prepare_call(args, kwargs)
        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = _run_awaitable(result)
            return self._format(result)
        except Exception as exc:
            raise ToolException(f"{type(exc).__name__}: {exc}") from exc

    async def _arun(
        self,
        *args: Any,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str | tuple[str, Any]:
        try:
            func = self.func
        except AttributeError:
            return f"Invalid function name: {self.name}"

        args, kwargs = self._prepare_call(args, kwargs)
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # Don't block the event loop on a synchronous SDK call.
                result = await asyncio.to_thread(func, *args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            return self._format(result)
        except Exception as exc:
            raise ToolException(f"{type(exc).__name__}: {exc}") from exc


def _run_awaitable(awaitable: Any) -> Any:
    """Resolve an awaitable from synchronous code.

    ``asyncio.run`` refuses to run inside an existing event loop, which is
    exactly where a synchronous ``invoke`` on an async SDK method tends to be
    called from. Fall back to a short-lived loop on a worker thread there.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await(awaitable))

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _await(awaitable)).result()


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

    fixed_args: dict[str, Any] = {}
    """Arguments pinned for every operation that accepts them, hidden from the model."""

    describe: DescriptionStyle | Callable[[Any, str], str] = "full"
    """``"full"``, ``"summary"`` (leading paragraph), or a ``(func, name)`` callable."""

    max_description_length: int | None = None
    """Hard ceiling on description length, applied after ``describe``."""

    response_format: Literal["content", "content_and_artifact"] = "content"
    """``"content_and_artifact"`` also returns the raw SDK result on the ToolMessage."""

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

            operations.append(
                AutoTool.from_client(
                    self.client,
                    func_name,
                    fixed_args=self.fixed_args,
                    describe=self.describe,
                    max_description_length=self.max_description_length,
                    response_format=self.response_format,
                )
            )

        return operations

    def get_tools(self) -> list[BaseTool]:
        """Get the tools in the toolkit."""
        return list(self.operations)
