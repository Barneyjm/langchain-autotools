import json

from langchain_autotools import AutoToolWrapper, CrudControls


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
    crud_controls=crud_controls,  # type: ignore
)


list_client = {"client": FakeSdk()}
list_crud_controls: CrudControls = CrudControls(
    read=True,
    read_list= ["get_thing"],
    create=False,
    update=False,
    delete=False,
)

list_autotool = AutoToolWrapper(
    client=list_client,
    crud_controls=list_crud_controls,  # type: ignore
)


def test_operations_is_populated() -> None:
    assert len(autotool.operations) != 0


def test_get_things_no_input() -> None:
    matching_tool = next(
        (tool for tool in autotool.operations if tool.name == "get_things"), None
    )
    if matching_tool:
        assert json.loads(matching_tool._run())["response"]["id"] == 123
    else:
        raise ValueError("No matching tool found.")


def test_get_thing() -> None:
    matching_tool = next(
        (tool for tool in autotool.operations if tool.name == "get_thing"), None
    )
    if matching_tool:
        assert (
            json.loads(matching_tool._run(json.dumps({"thing_id": 123})))["response"][
                "id"
            ]
            == 123
        )
    else:
        raise ValueError("No matching tool found.")
    
def test_only_get_thing() -> None:
    if len(list_autotool.operations) != 1:
        raise ValueError(f"Too many AutoTools, operations should only have `get_thing`, got: {[tool.name for tool in list_autotool.operations]}.")
    matching_tool = next(
        (tool for tool in list_autotool.operations if tool.name == "get_thing"), None
    )
    if matching_tool:
        assert (
            json.loads(matching_tool._run(json.dumps({"thing_id": 123})))["response"][
                "id"
            ]
            == 123
        )
    else:
        raise ValueError("No matching tool found.")


def test_create_thing_string_input() -> None:
    matching_tool = next(
        (tool for tool in autotool.operations if tool.name == "create_thing"), None
    )
    if matching_tool:
        assert (
            json.loads(matching_tool._run(json.dumps({"thing_id": 123})))["response"][
                "id"
            ]
            == 123
        )
    else:
        raise ValueError("No matching tool found.")


def test_create_thing_kwarg_input() -> None:
    matching_tool = next(
        (tool for tool in autotool.operations if tool.name == "create_thing"), None
    )
    if matching_tool:
        assert json.loads(matching_tool._run(thing_id=123))["response"]["id"] == 123  # type: ignore
    else:
        raise ValueError("No matching tool found.")

def test_update_thing() -> None:
    matching_tool = next(
        (tool for tool in autotool.operations if tool.name == "update_thing"), None
    )
    if matching_tool:
        assert (
            json.loads(matching_tool._run(json.dumps({"thing_id": 123})))["response"][
                "id"
            ]
            == 123
        )
    else:
        raise ValueError("No matching tool found.")

def test_post_thing() -> None:
    matching_tool = next(
        (tool for tool in autotool.operations if tool.name == "post_thing"), None
    )
    if matching_tool:
        assert (
            json.loads(matching_tool._run(json.dumps({"thing_id": 123})))["response"][
                "id"
            ]
            == 123
        )
    else:
        raise ValueError("No matching tool found.")


def test_put_thing() -> None:
    matching_tool = next(
        (tool for tool in autotool.operations if tool.name == "put_thing"), None
    )
    if matching_tool:
        assert (
            json.loads(matching_tool._run(json.dumps({"thing_id": 123})))["response"][
                "id"
            ]
            == 123
        )
    else:
        raise ValueError("No matching tool found.")


def test_delete_thing() -> None:
    matching_tool = next(
        (tool for tool in autotool.operations if tool.name == "delete_thing"), None
    )
    if matching_tool:
        assert (
            json.loads(matching_tool.run(json.dumps({"thing_id": 123})))["response"][
                "id"
            ]
            == 123
        )
    else:
        raise ValueError("No matching tool found.")


def test_confabulate_thing() -> None:
    """tests example sdk customization"""
    crud_controls = CrudControls(
        read_list=["confabulate_thing"],
    )

    autotool = AutoToolWrapper(
        client=client,
        crud_controls=crud_controls,  # type: ignore
    )
    matching_tool = next(
        (tool for tool in autotool.operations if tool.name == "confabulate_thing"), None
    )
    if matching_tool:
        assert (
            json.loads(matching_tool.run(json.dumps({"thing_id": 123})))["response"][
                "id"
            ]
            == 123
        )
    else:
        raise ValueError("No matching tool found.")

def test_generate_things() -> None:
    matching_tool = next(
        (tool for tool in autotool.operations if tool.name == "get_things_generator"), None
    )
    if matching_tool:
        result = json.loads(matching_tool._run(json.dumps({"start_id": 100, "count": 3})))
        
        assert hasattr(result, '__iter__')

        assert len(result) == 3

        for i, thing in enumerate(result):
            assert isinstance(thing, dict)
            assert thing["status"] == 200
            assert thing["response"]["id"] == 100 + i
    else:
        raise ValueError("No matching tool found.")

def test_no_hidden_methods() -> None:
    assert not any([op.name.startswith("_") for op in autotool.operations])