import json
import re
from fnmatch import fnmatch
from json import JSONDecodeError
from typing import Any, Optional, Union, List, Pattern
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, ConfigDict, model_validator
from langchain_core.tools import BaseTool
from collections.abc import Iterator, Iterable

# Default CRUD patterns using regex
AUTOTOOL_CRUD_CONTROLS_CREATE = False
AUTOTOOL_CRUD_CONTROLS_READ = True
AUTOTOOL_CRUD_CONTROLS_UPDATE = False
AUTOTOOL_CRUD_CONTROLS_DELETE = False

# Default patterns now support both regex (r"^pattern$") and glob ("pattern*") styles
AUTOTOOL_CRUD_CONTROLS_CREATE_LIST = [
    r"^create_[^_]+$",  # regex pattern
    "create_*",         # glob pattern (disabled by default)
]
AUTOTOOL_CRUD_CONTROLS_READ_LIST = [
    r"^get_[^_]+$",    # regex pattern
    "get_*",           # glob pattern (disabled by default)
]
AUTOTOOL_CRUD_CONTROLS_UPDATE_LIST = [
    r"^update_[^_]+$", # regex pattern
    "update_*",        # glob pattern (disabled by default)
]
AUTOTOOL_CRUD_CONTROLS_DELETE_LIST = [
    r"^delete_[^_]+$", # regex pattern
    "delete_*",        # glob pattern (disabled by default)
]

class CrudControls(BaseModel):
    create: Optional[bool] = AUTOTOOL_CRUD_CONTROLS_CREATE
    create_list: Optional[List[str]] = AUTOTOOL_CRUD_CONTROLS_CREATE_LIST
    read: Optional[bool] = AUTOTOOL_CRUD_CONTROLS_READ
    read_list: Optional[List[str]] = AUTOTOOL_CRUD_CONTROLS_READ_LIST
    update: Optional[bool] = AUTOTOOL_CRUD_CONTROLS_UPDATE
    update_list: Optional[List[str]] = AUTOTOOL_CRUD_CONTROLS_UPDATE_LIST
    delete: Optional[bool] = AUTOTOOL_CRUD_CONTROLS_DELETE
    delete_list: Optional[List[str]] = AUTOTOOL_CRUD_CONTROLS_DELETE_LIST
    _compiled_patterns: dict[str, List[Pattern]] = {}

    @model_validator(mode="before")
    def validate_environment(cls, values: dict) -> "CrudControls":
        for crud_type in ['create', 'read', 'update', 'delete']:
            values[crud_type] = values.get(crud_type, globals()[f"AUTOTOOL_CRUD_CONTROLS_{crud_type.upper()}"])
            values[f"{crud_type}_list"] = values.get(f"{crud_type}_list", 
                globals()[f"AUTOTOOL_CRUD_CONTROLS_{crud_type.upper()}_LIST"])
        return values

    def _is_regex_pattern(self, pattern: str) -> bool:
        """Check if a pattern is a regex pattern (starts with r or contains regex special chars)."""
        regex_chars = '.^$*+?{}[]|\\()'
        return pattern.startswith('r"') or pattern.startswith("r'") or any(c in pattern for c in regex_chars)

    def compile_patterns(self) -> None:
        """Compile regex patterns and store glob patterns for each CRUD operation type."""
        self._compiled_patterns.clear()
        
        for crud_type in ['create', 'read', 'update', 'delete']:
            pattern_list = getattr(self, f"{crud_type}_list", [])
            regex_patterns = []
            glob_patterns = []
            
            for pattern in pattern_list:
                if isinstance(pattern, str):
                    if self._is_regex_pattern(pattern):
                        # Strip 'r' prefix and quotes if present
                        if pattern.startswith('r"') or pattern.startswith("r'"):
                            pattern = pattern[2:-1]
                        try:
                            regex_patterns.append(re.compile(pattern))
                        except re.error:
                            # If regex compilation fails, treat as glob pattern
                            glob_patterns.append(pattern)
                    else:
                        glob_patterns.append(pattern)
            
            self._compiled_patterns[crud_type] = {
                'regex': regex_patterns,
                'glob': glob_patterns
            }

    def matches_pattern(self, func_name: str, crud_type: str) -> bool:
        """Check if a function name matches any pattern (regex or glob) for a given CRUD type."""
        if not self._compiled_patterns:
            self.compile_patterns()
            
        if not getattr(self, crud_type, False):
            return False
            
        patterns = self._compiled_patterns.get(crud_type, {'regex': [], 'glob': []})
        
        # Check regex patterns
        if any(pattern.match(func_name) for pattern in patterns['regex']):
            return True
            
        # Check glob patterns
        if any(fnmatch(func_name, pattern) for pattern in patterns['glob']):
            return True
            
        return False


class AutoTool(BaseTool):
    """Tool for whatever function is passed into AutoTool."""

    client: Any
    name: str
    description: str

    def _run(
        self,
        tool_input: Union[str, dict, None] = None,
        *args: tuple,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: dict,
    ) -> str:
        try:
            if isinstance(tool_input, dict):
                params = tool_input
            elif isinstance(tool_input, str):
                try:
                    params = json.loads(tool_input)
                except JSONDecodeError:
                    params = {}
            else:
                params = {}

            func = getattr(self.client["client"], self.name)
            result = func(*args, **params, **kwargs)
            if isinstance(result, dict):
                result = result
            elif isinstance(result, (Iterator, Iterable)) and not isinstance(result, (str, bytes)):
                result = list(result)
            return json.dumps(result, default=str)
        except AttributeError:
            return f"Invalid function name: {self.name}"

    async def _arun(
        self,
        tool_input: Union[str, dict, None] = None,
        *args: tuple,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: dict,
    ) -> str:
        return self._run(
            tool_input,
            *args,
            run_manager=run_manager,
            **kwargs,
        )

class AutoToolWrapper(BaseModel):
    client: Any
    operations: List[AutoTool] = []
    crud_controls: CrudControls = CrudControls()
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow"
    )

    def __init__(self, **data: dict) -> None:
        super().__init__(**data)
        self.operations = self._build_operations()

    def _build_operations(self) -> list:
        operations: List[BaseTool] = []
        sdk_functions = [
            func
            for func in dir(self.client["client"])
            if (
                callable(getattr(self.client["client"], func))
                and not func.startswith("_")
            )
        ]

        for func_name in sdk_functions:
            func = getattr(self.client["client"], func_name)
            operation = AutoTool(
                client=self.client, name=func_name, description=func.__doc__
            )
            
            # Check if function matches any CRUD pattern
            should_add = any(
                self.crud_controls.matches_pattern(func_name, crud_type)
                for crud_type in ['create', 'read', 'update', 'delete']
            )
            
            if should_add:
                operations.append(operation)

        return operations

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [op for op in self.operations]