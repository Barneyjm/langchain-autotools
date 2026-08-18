import asyncio
import json

import pytest
from pydantic import ValidationError

from langchain_autotools import AutoTool, AutoToolWrapper, CrudControls


class FakeSdk:
    def get_things(self) -> dict:
        """Gets Thing"""
        return self._dummy_return(123)

    def get_thing(self, thing_id: int) -> dict:
        """Gets Thing"""
        return self._dummy_return(thing_id)

    def create_thing(self, thing_id: int) -> dict:
        """Creates Thing"""
        return self._dummy_return(thing_id)

    def post_thing(self, thing_id: int) -> dict:
        """Posts Thing"""
        return self._dummy_return(thing_id)

    def put_thing(self, thing_id: int) -> dict:
        """Puts Thing"""
        return self._dummy_return(thing_id)

    def update_thing(self, thing_id: int) -> dict:
        """Updates Thing"""
        return self._dummy_return(thing_id)

    def delete_thing(self, thing_id: int) -> dict:
        """Deletes Thing"""
        return self._dummy_return(thing_id)

    def confabulate_thing(self, thing_id: int) -> dict:
        """Confabulates Thing -- example of custom verbs"""
        return self._dummy_return(thing_id)

    def get_things_generator(self, start_id: int, count: int):
        """Generates multiple Things"""
        for i in range(count):
            yield self._dummy_return(start_id + i)

    def get_undocumented(self, thing_id: int = 1) -> dict:
        return self._dummy_return(thing_id)

    async def get_thing_async(self, thing_id: int) -> dict:
        """Gets Thing, asynchronously"""
        await asyncio.sleep(0)
        return self._dummy_return(thing_id)

    def get_dynamic(self, *args, **kwargs) -> dict:
        """Boto3-style dynamic operation"""
        return {"status": 200, "response": kwargs}

    def _dummy_return(self, thing_id: int) -> dict:
        """Hidden, never called directly. Does Things"""
        return {"status": 200, "response": {"id": thing_id}}


client = {"client": FakeSdk()}
crud_controls: CrudControls = CrudControls(
    read=True,
    create=True,
    update=True,
    update_list=["put_thing", "post_thing", "update_thing"],
    delete=True,
)

autotool = AutoToolWrapper(
    client=client,
    crud_controls=crud_controls,
)


list_client = {"client": FakeSdk()}
list_crud_controls: CrudControls = CrudControls(
    read=True,
    read_list=["get_thing"],
    create=False,
    update=False,
    delete=False,
)

list_autotool = AutoToolWrapper(
    client=list_client,
    crud_controls=list_crud_controls,
)


def tool_named(name: str, toolkit: AutoToolWrapper = autotool) -> AutoTool:
    matching_tool = next(
        (tool for tool in toolkit.operations if tool.name == name), None
    )
    if matching_tool is None:
        raise ValueError(f"No matching tool found for {name!r}.")
    return matching_tool


def test_operations_is_populated() -> None:
    assert len(autotool.operations) != 0


def test_get_tools_matches_operations() -> None:
    assert [tool.name for tool in autotool.get_tools()] == [
        tool.name for tool in autotool.operations
    ]


def test_get_things_no_input() -> None:
    assert json.loads(tool_named("get_things").invoke({}))["response"]["id"] == 123


def test_get_thing() -> None:
    result = json.loads(tool_named("get_thing").invoke({"thing_id": 123}))
    assert result["response"]["id"] == 123


def test_only_get_thing() -> None:
    assert [tool.name for tool in list_autotool.operations] == ["get_thing"]
    result = json.loads(
        tool_named("get_thing", list_autotool).invoke({"thing_id": 123})
    )
    assert result["response"]["id"] == 123


def test_create_thing() -> None:
    result = json.loads(tool_named("create_thing").invoke({"thing_id": 123}))
    assert result["response"]["id"] == 123


def test_update_thing() -> None:
    result = json.loads(tool_named("update_thing").invoke({"thing_id": 123}))
    assert result["response"]["id"] == 123


def test_post_thing() -> None:
    result = json.loads(tool_named("post_thing").invoke({"thing_id": 123}))
    assert result["response"]["id"] == 123


def test_put_thing() -> None:
    result = json.loads(tool_named("put_thing").invoke({"thing_id": 123}))
    assert result["response"]["id"] == 123


def test_delete_thing() -> None:
    result = json.loads(tool_named("delete_thing").invoke({"thing_id": 123}))
    assert result["response"]["id"] == 123


def test_confabulate_thing() -> None:
    """tests example sdk customization"""
    toolkit = AutoToolWrapper(
        client=client,
        crud_controls=CrudControls(read_list=["confabulate_thing"]),
    )
    result = json.loads(
        tool_named("confabulate_thing", toolkit).invoke({"thing_id": 123})
    )
    assert result["response"]["id"] == 123


def test_generate_things() -> None:
    """Generators are drained so the result is JSON serializable."""
    result = json.loads(
        tool_named("get_things_generator").invoke({"start_id": 100, "count": 3})
    )
    assert len(result) == 3
    for i, thing in enumerate(result):
        assert thing["status"] == 200
        assert thing["response"]["id"] == 100 + i


def test_no_hidden_methods() -> None:
    assert not any(op.name.startswith("_") for op in autotool.operations)


# --- args schema -----------------------------------------------------------


def test_args_schema_is_derived_from_signature() -> None:
    schema = tool_named("get_thing").args_schema.model_json_schema()
    assert schema["properties"]["thing_id"]["type"] == "integer"
    assert schema["required"] == ["thing_id"]


def test_no_arg_function_has_empty_schema() -> None:
    assert tool_named("get_things").args_schema.model_json_schema()["properties"] == {}


def test_invalid_argument_is_rejected() -> None:
    with pytest.raises(ValidationError):
        tool_named("get_thing").invoke({"thing_id": "not-an-int"})


def test_dynamic_signature_passes_kwargs_through() -> None:
    """``(*args, **kwargs)`` clients keep working without an invented schema."""
    tool = tool_named("get_dynamic")
    assert tool.args_schema is None
    result = json.loads(tool.invoke({"Bucket": "mybucket", "Key": "k"}))
    assert result["response"] == {"Bucket": "mybucket", "Key": "k"}


def test_description_falls_back_to_signature() -> None:
    description = tool_named("get_undocumented").description
    assert "get_undocumented" in description
    assert description


def test_description_uses_docstring() -> None:
    assert tool_named("get_thing").description == "Gets Thing"


# --- async -----------------------------------------------------------------


def test_async_sdk_function() -> None:
    tool = tool_named("get_thing_async")
    result = json.loads(asyncio.run(tool.ainvoke({"thing_id": 5})))
    assert result["response"]["id"] == 5


def test_sync_function_over_async_interface() -> None:
    result = json.loads(asyncio.run(tool_named("get_thing").ainvoke({"thing_id": 5})))
    assert result["response"]["id"] == 5


def test_async_function_over_sync_interface() -> None:
    result = json.loads(tool_named("get_thing_async").invoke({"thing_id": 5}))
    assert result["response"]["id"] == 5


# --- client forms ----------------------------------------------------------


def test_bare_client_is_accepted() -> None:
    toolkit = AutoToolWrapper(client=FakeSdk())
    assert "get_thing" in [tool.name for tool in toolkit.get_tools()]


def test_legacy_dict_client_is_accepted() -> None:
    toolkit = AutoToolWrapper(client={"client": FakeSdk()})
    assert "get_thing" in [tool.name for tool in toolkit.get_tools()]


def test_legacy_json_payload_still_runs() -> None:
    """Direct ``_run`` calls with a single JSON payload are still supported."""
    tool = tool_named("get_thing")
    assert json.loads(tool._run(json.dumps({"thing_id": 123})))["response"]["id"] == 123
    assert json.loads(tool._run(thing_id=123))["response"]["id"] == 123


def test_invalid_function_name() -> None:
    tool = AutoTool(client=FakeSdk(), name="nope", description="nope")
    assert tool._run() == "Invalid function name: nope"


# --- crud controls ---------------------------------------------------------


def test_glob_pattern_matches_prefix() -> None:
    controls = CrudControls(read=True, read_list=["get_thing*"])
    assert controls.matches_pattern("get_thing", "read")
    assert controls.matches_pattern("get_thing_by_id", "read")
    assert not controls.matches_pattern("create_thing", "read")


def test_regex_pattern_is_detected() -> None:
    controls = CrudControls(read=True, read_list=[r"^get_thing_\w+$"])
    assert controls.matches_pattern("get_thing_by_id", "read")
    assert not controls.matches_pattern("get_thing", "read")


def test_glob_and_regex_can_be_mixed() -> None:
    controls = CrudControls(read=True, read_list=["get_thing*", r"^list_\w+$"])
    assert controls.matches_pattern("get_thing_by_id", "read")
    assert controls.matches_pattern("list_buckets", "read")
    assert not controls.matches_pattern("delete_thing", "read")


def test_disabled_verb_matches_nothing() -> None:
    controls = CrudControls(delete=False, delete_list=["delete_*"])
    assert not controls.matches_pattern("delete_thing", "delete")


def test_controls_do_not_share_compiled_patterns() -> None:
    """Each instance compiles its own patterns."""
    first = CrudControls(read=True, read_list=["get_thing"])
    second = CrudControls(read=True, read_list=["list_*"])
    assert first.matches_pattern("get_thing", "read")
    assert not first.matches_pattern("list_buckets", "read")
    assert second.matches_pattern("list_buckets", "read")
    assert not second.matches_pattern("get_thing", "read")


def test_defaults_are_read_only() -> None:
    controls = CrudControls()
    assert controls.allows("get_thing")
    assert not controls.allows("create_thing")
    assert not controls.allows("delete_thing")


# --- tool calling ----------------------------------------------------------


def test_tool_converts_to_tool_calling_schema() -> None:
    """The generated tools are usable by modern tool-calling models."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    spec = convert_to_openai_tool(tool_named("get_thing"))
    assert spec["function"]["name"] == "get_thing"
    assert spec["function"]["description"] == "Gets Thing"
    assert spec["function"]["parameters"]["properties"]["thing_id"]["type"] == "integer"
