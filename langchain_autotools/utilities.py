from typing import Any, List, Optional

from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.tools import BaseTool

from tool import AutoTool

AUTOTOOL_CRUD_CONTROLS_CREATE = False
AUTOTOOL_CRUD_CONTROLS_READ = True
AUTOTOOL_CRUD_CONTROLS_UPDATE = False
AUTOTOOL_CRUD_CONTROLS_DELETE = False

AUTOTOOL_CRUD_CONTROLS_CREATE_LIST = "create"
AUTOTOOL_CRUD_CONTROLS_READ_LIST = "get,read,list"
AUTOTOOL_CRUD_CONTROLS_UPDATE_LIST = "update,put,post"
AUTOTOOL_CRUD_CONTROLS_DELETE_LIST = "delete,destroy,remove"


class CrudControls(BaseModel):
    create: Optional[bool] = AUTOTOOL_CRUD_CONTROLS_CREATE
    create_list: Optional[str] = AUTOTOOL_CRUD_CONTROLS_CREATE_LIST
    read: Optional[bool] = AUTOTOOL_CRUD_CONTROLS_READ
    read_list: Optional[str] = AUTOTOOL_CRUD_CONTROLS_READ_LIST
    update: Optional[bool] = AUTOTOOL_CRUD_CONTROLS_UPDATE
    update_list: Optional[str] = AUTOTOOL_CRUD_CONTROLS_UPDATE_LIST
    delete: Optional[bool] = AUTOTOOL_CRUD_CONTROLS_DELETE
    delete_list: Optional[str] = AUTOTOOL_CRUD_CONTROLS_DELETE_LIST

    @root_validator
    def validate_environment(cls, values: dict) -> "CrudControls":
        create = values.get("create", AUTOTOOL_CRUD_CONTROLS_CREATE)
        values["create"] = create

        create_list = values.get("create_list", AUTOTOOL_CRUD_CONTROLS_CREATE_LIST)
        values["create_list"] = create_list.split(",")

        read = values.get("read", AUTOTOOL_CRUD_CONTROLS_READ)
        values["read"] = read

        read_list = values.get("read_list", AUTOTOOL_CRUD_CONTROLS_READ_LIST)
        values["read_list"] = read_list.split(",")

        update = values.get("update", AUTOTOOL_CRUD_CONTROLS_UPDATE)
        values["update"] = update

        update_list = values.get("update_list", AUTOTOOL_CRUD_CONTROLS_UPDATE_LIST)
        values["update_list"] = update_list.split(",")

        delete = values.get("delete", AUTOTOOL_CRUD_CONTROLS_DELETE)
        values["delete"] = delete

        delete_list = values.get("delete_list", AUTOTOOL_CRUD_CONTROLS_DELETE_LIST)
        values["delete_list"] = delete_list.split(",")

        return values  # type: ignore


class AutoToolWrapper(BaseModel):
    client: Any
    operations: List[AutoTool] = []
    crud_controls: CrudControls = CrudControls()

    class Config:
        extra = Extra.forbid

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
            if self.crud_controls:
                if self.crud_controls.create:
                    if self.crud_controls.create_list is not None and any(
                        word.lower() in func_name.lower()
                        for word in self.crud_controls.create_list
                    ):
                        operations.append(operation)

                if self.crud_controls.read:
                    if self.crud_controls.read_list is not None and any(
                        word.lower() in func_name.lower()
                        for word in self.crud_controls.read_list
                    ):
                        operations.append(operation)

                if self.crud_controls.update:
                    if self.crud_controls.update_list is not None and any(
                        word.lower() in func_name.lower()
                        for word in self.crud_controls.update_list
                    ):
                        operations.append(operation)

                if self.crud_controls.delete:
                    if self.crud_controls.delete_list is not None and any(
                        word.lower() in func_name.lower()
                        for word in self.crud_controls.delete_list
                    ):
                        operations.append(operation)

        return operations

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [op for op in self.operations]