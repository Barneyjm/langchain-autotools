import asyncio
import json

import pytest
from langchain_core.tools import ToolException
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


def test_async_function_over_sync_interface_inside_loop() -> None:
    """A sync ``invoke`` on an async SDK method works inside a running loop."""

    async def main() -> str:
        return tool_named("get_thing_async").invoke({"thing_id": 5})

    assert json.loads(asyncio.run(main()))["response"]["id"] == 5


# --- error handling --------------------------------------------------------


class BoomSdk:
    def get_boom(self, thing_id: int) -> dict:
        """Raises the way a real SDK does"""
        raise PermissionError("AccessDenied: not authorized")


def test_sdk_error_is_returned_to_the_agent() -> None:
    """A failing SDK call becomes a tool result, not the end of the run."""
    toolkit = AutoToolWrapper(client=BoomSdk())
    result = tool_named("get_boom", toolkit).invoke({"thing_id": 1})
    assert result == "PermissionError: AccessDenied: not authorized"


def test_sdk_error_can_propagate() -> None:
    """handle_tool_error=False re-raises, keeping the original as __cause__."""
    toolkit = AutoToolWrapper(client=BoomSdk())
    tool = tool_named("get_boom", toolkit)
    tool.handle_tool_error = False
    with pytest.raises(ToolException) as excinfo:
        tool.invoke({"thing_id": 1})
    assert isinstance(excinfo.value.__cause__, PermissionError)


def test_missing_method_is_not_wrapped() -> None:
    tool = AutoTool(client=FakeSdk(), name="nope", description="nope")
    assert tool._run() == "Invalid function name: nope"


# --- pinned arguments ------------------------------------------------------


class TenantSdk:
    def get_thing(self, thing_id: int, tenant: str = "default") -> dict:
        """Gets Thing"""
        return {"id": thing_id, "tenant": tenant}

    def get_global(self, thing_id: int) -> dict:
        """Takes no tenant"""
        return {"id": thing_id}

    def get_dynamic(self, **kwargs) -> dict:
        """Accepts anything"""
        return dict(kwargs)


def test_fixed_args_are_hidden_from_the_schema() -> None:
    toolkit = AutoToolWrapper(client=TenantSdk(), fixed_args={"tenant": "acme"})
    schema = tool_named("get_thing", toolkit).args_schema.model_json_schema()
    assert list(schema["properties"]) == ["thing_id"]


def test_fixed_args_are_applied() -> None:
    toolkit = AutoToolWrapper(client=TenantSdk(), fixed_args={"tenant": "acme"})
    result = json.loads(tool_named("get_thing", toolkit).invoke({"thing_id": 1}))
    assert result["tenant"] == "acme"


def test_fixed_args_win_over_model_supplied_values() -> None:
    toolkit = AutoToolWrapper(client=TenantSdk(), fixed_args={"tenant": "acme"})
    tool = tool_named("get_thing", toolkit)
    result = json.loads(tool._run(thing_id=1, tenant="evil"))
    assert result["tenant"] == "acme"


def test_fixed_args_skip_functions_that_cannot_accept_them() -> None:
    """A toolkit-wide pin must not break operations without that parameter."""
    toolkit = AutoToolWrapper(client=TenantSdk(), fixed_args={"tenant": "acme"})
    assert tool_named("get_global", toolkit).fixed_args == {}
    assert json.loads(tool_named("get_global", toolkit).invoke({"thing_id": 1})) == {
        "id": 1
    }


def test_fixed_args_reach_kwargs_catchalls() -> None:
    toolkit = AutoToolWrapper(client=TenantSdk(), fixed_args={"tenant": "acme"})
    result = json.loads(tool_named("get_dynamic", toolkit).invoke({"thing_id": 1}))
    assert result == {"thing_id": 1, "tenant": "acme"}


# --- descriptions ----------------------------------------------------------


class DocSdk:
    def get_documented(self, thing_id: int) -> dict:
        """Gets Thing.

        A second paragraph with pagination trivia that a model does not need,
        since the argument schema already describes the call.
        """
        return {"id": thing_id}

    def get_dynamic(self, *args, **kwargs) -> dict:
        """Gets Thing dynamically.

        Parameters are documented only here: pass thing_id.
        """
        return {}


def test_summary_uses_the_leading_paragraph() -> None:
    toolkit = AutoToolWrapper(client=DocSdk(), describe="summary")
    assert tool_named("get_documented", toolkit).description == "Gets Thing."


def test_summary_applies_without_a_schema() -> None:
    """Summary trims dynamic signatures too, losing their only arg reference."""
    toolkit = AutoToolWrapper(client=DocSdk(), describe="summary")
    assert tool_named("get_dynamic", toolkit).description == "Gets Thing dynamically."


def test_max_description_length_always_applies() -> None:
    toolkit = AutoToolWrapper(client=DocSdk(), max_description_length=20)
    for tool in toolkit.get_tools():
        assert len(tool.description) <= 23  # 20 plus the ellipsis
    assert tool_named("get_dynamic", toolkit).description.endswith("...")


def test_describe_accepts_a_callable() -> None:
    toolkit = AutoToolWrapper(
        client=DocSdk(), describe=lambda func, name: f"call {name}"
    )
    assert tool_named("get_dynamic", toolkit).description == "call get_dynamic"


# --- artifacts -------------------------------------------------------------


def test_content_and_artifact_returns_the_raw_result() -> None:
    toolkit = AutoToolWrapper(client=FakeSdk(), response_format="content_and_artifact")
    tool = tool_named("get_thing", toolkit)
    message = tool.invoke(
        {"name": "get_thing", "args": {"thing_id": 3}, "id": "1", "type": "tool_call"}
    )
    assert message.content == json.dumps({"status": 200, "response": {"id": 3}})
    assert message.artifact == {"status": 200, "response": {"id": 3}}


def test_content_and_artifact_drains_generators_once() -> None:
    toolkit = AutoToolWrapper(client=FakeSdk(), response_format="content_and_artifact")
    tool = tool_named("get_things_generator", toolkit)
    message = tool.invoke(
        {
            "name": "get_things_generator",
            "args": {"start_id": 100, "count": 3},
            "id": "1",
            "type": "tool_call",
        }
    )
    assert isinstance(message.artifact, list)
    assert len(message.artifact) == 3
    assert json.loads(message.content) == message.artifact


def test_default_response_format_is_a_string() -> None:
    assert isinstance(tool_named("get_thing").invoke({"thing_id": 1}), str)


# --- exclude patterns ------------------------------------------------------


def test_exclude_vetoes_a_wildcard_match() -> None:
    """Wildcards stay broad; exclude drops the individual bad matches."""
    controls = CrudControls(
        read=True, read_list=["get_*"], exclude=["get_paginator", "get_waiter"]
    )
    assert controls.allows("get_object")
    assert not controls.allows("get_paginator")
    assert not controls.allows("get_waiter")


def test_exclude_accepts_globs_and_regexes() -> None:
    controls = CrudControls(read=True, read_list=["get_*"], exclude=["*_internal"])
    assert not controls.allows("get_thing_internal")
    assert controls.allows("get_thing")

    controls = CrudControls(read=True, read_list=["get_*"], exclude=[r"^get_\d+$"])
    assert not controls.allows("get_1")
    assert controls.allows("get_thing")


def test_exclude_keeps_operations_out_of_the_toolkit() -> None:
    toolkit = AutoToolWrapper(
        client=FakeSdk(),
        crud_controls=CrudControls(
            read=True, read_list=["get_*"], exclude=["get_things*"]
        ),
    )
    names = [tool.name for tool in toolkit.get_tools()]
    assert "get_thing" in names
    assert "get_things" not in names
    assert "get_things_generator" not in names


def test_no_exclude_by_default() -> None:
    assert CrudControls().allows("get_thing")


# --- crud metadata ---------------------------------------------------------


def test_tools_record_the_matched_verb() -> None:
    toolkit = AutoToolWrapper(client=FakeSdk(), crud_controls=crud_controls)
    verbs = {tool.name: tool.metadata["crud"] for tool in toolkit.get_tools()}
    assert verbs["get_thing"] == "read"
    assert verbs["create_thing"] == "create"
    assert verbs["update_thing"] == "update"
    assert verbs["delete_thing"] == "delete"


def test_metadata_records_the_sdk_function() -> None:
    toolkit = AutoToolWrapper(client=FakeSdk(), prefix="fake_")
    tool = tool_named("fake_get_thing", toolkit)
    assert tool.metadata["sdk_function"] == "get_thing"


def test_matched_verb_is_stable_across_overlapping_lists() -> None:
    controls = CrudControls(
        read=True, read_list=["get_*"], delete=True, delete_list=["get_*"]
    )
    assert controls.matched_verb("get_thing") == "read"


def test_destructive_tools_can_be_selected() -> None:
    """The use case the metadata exists for: gating write operations."""
    toolkit = AutoToolWrapper(client=FakeSdk(), crud_controls=crud_controls)
    destructive = [
        tool.name
        for tool in toolkit.get_tools()
        if tool.metadata["crud"] in ("create", "update", "delete")
    ]
    assert set(destructive) == {
        "create_thing",
        "update_thing",
        "post_thing",
        "put_thing",
        "delete_thing",
    }


# --- name prefix -----------------------------------------------------------


def test_prefix_renames_tools_without_breaking_dispatch() -> None:
    toolkit = AutoToolWrapper(client=FakeSdk(), prefix="fake_")
    tool = tool_named("fake_get_thing", toolkit)
    assert tool.func_name == "get_thing"
    assert json.loads(tool.invoke({"thing_id": 7}))["response"]["id"] == 7


def test_prefix_keeps_two_sdks_apart() -> None:
    first = AutoToolWrapper(client=FakeSdk(), prefix="a_")
    second = AutoToolWrapper(client=FakeSdk(), prefix="b_")
    names = [t.name for t in first.get_tools()] + [t.name for t in second.get_tools()]
    assert len(names) == len(set(names))


def test_prefixed_missing_method_names_the_sdk_function() -> None:
    tool = AutoTool(
        client=FakeSdk(), name="fake_nope", func_name="nope", description="nope"
    )
    assert tool._run() == "Invalid function name: nope"


# --- result truncation -----------------------------------------------------


class BulkSdk:
    def get_keys(self, count: int) -> list:
        """Lists keys."""
        return [{"Key": f"part-{i:05d}"} for i in range(count)]

    def get_blob(self, size: int) -> dict:
        """Gets one large record."""
        return {"body": "x" * size}


def test_list_results_truncate_on_an_item_boundary() -> None:
    toolkit = AutoToolWrapper(client=BulkSdk(), max_result_length=300)
    content = tool_named("get_keys", toolkit).invoke({"count": 1000})
    body, marker = content.split("\n\n")
    assert isinstance(json.loads(body), list)  # kept portion is still valid JSON
    assert "of 1,000 items omitted" in marker


def test_non_list_results_truncate_on_a_character_boundary() -> None:
    toolkit = AutoToolWrapper(client=BulkSdk(), max_result_length=200)
    content = tool_named("get_blob", toolkit).invoke({"size": 50_000})
    assert "characters omitted" in content
    assert len(content) < 400


def test_results_under_the_limit_are_untouched() -> None:
    toolkit = AutoToolWrapper(client=BulkSdk(), max_result_length=10_000)
    content = tool_named("get_keys", toolkit).invoke({"count": 3})
    assert json.loads(content) == [{"Key": f"part-{i:05d}"} for i in range(3)]


def test_no_truncation_by_default() -> None:
    toolkit = AutoToolWrapper(client=BulkSdk())
    content = tool_named("get_keys", toolkit).invoke({"count": 500})
    assert len(json.loads(content)) == 500


def test_truncation_leaves_the_artifact_whole() -> None:
    """Truncation protects the context window, not the calling code."""
    toolkit = AutoToolWrapper(
        client=BulkSdk(), max_result_length=200, response_format="content_and_artifact"
    )
    message = tool_named("get_keys", toolkit).invoke(
        {"name": "get_keys", "args": {"count": 500}, "id": "1", "type": "tool_call"}
    )
    assert "items omitted" in message.content
    assert len(message.artifact) == 500


# --- argument descriptions -------------------------------------------------


class DocumentedSdk:
    def get_google(self, thing_id: int, verbose: bool = False) -> dict:
        """Gets a thing.

        Args:
            thing_id: Identifier of the thing,
                as issued by the registry.
            verbose (bool): Include the full record.
        """
        return {}

    def get_sphinx(self, thing_id: int) -> dict:
        """Gets a thing.

        :param int thing_id: Identifier of the thing.
        """
        return {}

    def get_undocumented_params(self, thing_id: int) -> dict:
        """No parameter docs here."""
        return {}


def _properties(toolkit: AutoToolWrapper, name: str) -> dict:
    return tool_named(name, toolkit).args_schema.model_json_schema()["properties"]


def test_google_style_arg_docs_reach_the_schema() -> None:
    props = _properties(AutoToolWrapper(client=DocumentedSdk()), "get_google")
    assert props["thing_id"]["description"] == (
        "Identifier of the thing, as issued by the registry."
    )
    assert props["verbose"]["description"] == "Include the full record."


def test_sphinx_style_arg_docs_reach_the_schema() -> None:
    props = _properties(AutoToolWrapper(client=DocumentedSdk()), "get_sphinx")
    assert props["thing_id"]["description"] == "Identifier of the thing."


def test_missing_arg_docs_are_not_invented() -> None:
    toolkit = AutoToolWrapper(client=DocumentedSdk())
    props = _properties(toolkit, "get_undocumented_params")
    assert "description" not in props["thing_id"]


def test_arg_docs_survive_summarised_descriptions() -> None:
    """Summary trims the description; the schema keeps the argument detail."""
    toolkit = AutoToolWrapper(client=DocumentedSdk(), describe="summary")
    assert tool_named("get_google", toolkit).description == "Gets a thing."
    assert _properties(toolkit, "get_google")["thing_id"]["description"]
