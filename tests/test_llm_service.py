import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from enterprise_ai_knowledge_assistant.services.llm_service import (
    MAX_LLM_CALLS,
    LLMService,
    LLMServiceError,
    execute_tool,
)


# ---- Fakes that mimic the shape of OpenRouter responses (no network) ----


def make_response(content: str | None = None, tool_calls: list | None = None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_tool_call(name: str, arguments: str, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeChat:
    """Returns the queued responses in order and records every request."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    async def send_async(self, **kwargs):
        # Copy, because the service keeps appending to the same messages list.
        self.requests.append(copy.deepcopy(kwargs))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_service(responses: list) -> tuple[LLMService, FakeChat]:
    service = LLMService()
    fake_chat = FakeChat(responses)
    service.client = SimpleNamespace(chat=fake_chat)
    return service, fake_chat


# ---- execute_tool: the backend side of tool calling ----


def test_execute_tool_runs_employee_lookup() -> None:
    result = execute_tool("get_employee_leave_balance", '{"employee_id": 101}')

    assert result == {"employee_id": 101, "name": "Palraj", "leave_balance": 8}


def test_execute_tool_with_missing_employee() -> None:
    result = execute_tool("get_employee_leave_balance", '{"employee_id": 999}')

    assert result == {"error": "Employee with id 999 not found"}


def test_execute_tool_with_unknown_tool() -> None:
    result = execute_tool("delete_all_employees", "{}")

    assert result == {"error": "Unknown tool: delete_all_employees"}


def test_execute_tool_with_wrong_argument_type() -> None:
    result = execute_tool("get_employee_leave_balance", '{"employee_id": "abc"}')

    assert result["error"].startswith("Invalid arguments for get_employee_leave_balance")


def test_execute_tool_with_missing_argument() -> None:
    result = execute_tool("get_employee_leave_balance", "{}")

    assert result["error"].startswith("Invalid arguments for get_employee_leave_balance")


def test_execute_tool_with_malformed_json() -> None:
    result = execute_tool("get_employee_leave_balance", '{"employee_id": 101')

    assert result["error"].startswith("Invalid arguments for get_employee_leave_balance")


# ---- generate_response: the full loop ----


def test_normal_answer_without_tool_call() -> None:
    service, fake_chat = make_service(
        [make_response(content="An AI agent is a system that acts on goals.")]
    )

    answer = asyncio.run(service.generate_response("What is an AI agent?"))

    assert answer == "An AI agent is a system that acts on goals."
    assert len(fake_chat.requests) == 1
    # The tool definitions are always offered; the LLM decides to not use them.
    tool_names = [tool["function"]["name"] for tool in fake_chat.requests[0]["tools"]]
    assert tool_names == ["get_employee_leave_balance"]


def test_tool_call_then_final_answer() -> None:
    service, fake_chat = make_service(
        [
            # 1st LLM reply: "please run this tool for me"
            make_response(
                tool_calls=[
                    make_tool_call("get_employee_leave_balance", '{"employee_id": 101}')
                ]
            ),
            # 2nd LLM reply: the final answer, written from the tool result
            make_response(content="Palraj has 8 leave days remaining."),
        ]
    )

    answer = asyncio.run(
        service.generate_response("How many leave days does employee 101 have?")
    )

    assert answer == "Palraj has 8 leave days remaining."
    assert len(fake_chat.requests) == 2

    # The 2nd request = system + user + assistant tool request + tool result.
    second_messages = fake_chat.requests[1]["messages"]
    assert [m["role"] for m in second_messages] == ["system", "user", "assistant", "tool"]

    assistant_message = second_messages[2]
    assert assistant_message["tool_calls"][0]["id"] == "call_1"
    assert assistant_message["tool_calls"][0]["function"]["name"] == "get_employee_leave_balance"

    tool_message = second_messages[3]
    assert tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"]) == {
        "employee_id": 101,
        "name": "Palraj",
        "leave_balance": 8,
    }


def test_tool_call_for_missing_employee_sends_error_to_llm() -> None:
    service, fake_chat = make_service(
        [
            make_response(
                tool_calls=[
                    make_tool_call("get_employee_leave_balance", '{"employee_id": 999}')
                ]
            ),
            make_response(content="I could not find employee 999."),
        ]
    )

    answer = asyncio.run(service.generate_response("Leave balance for employee 999?"))

    assert answer == "I could not find employee 999."
    tool_message = fake_chat.requests[1]["messages"][3]
    assert json.loads(tool_message["content"]) == {
        "error": "Employee with id 999 not found"
    }


def test_multiple_tool_calls_in_one_reply() -> None:
    service, fake_chat = make_service(
        [
            make_response(
                tool_calls=[
                    make_tool_call("get_employee_leave_balance", '{"employee_id": 101}', "call_a"),
                    make_tool_call("get_employee_leave_balance", '{"employee_id": 102}', "call_b"),
                ]
            ),
            make_response(content="Palraj has 8 days and John has 12 days."),
        ]
    )

    asyncio.run(service.generate_response("Compare employees 101 and 102."))

    tool_messages = [m for m in fake_chat.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_a", "call_b"]


def test_api_failure_raises_llm_service_error() -> None:
    service, _ = make_service([ConnectionError("network down")])

    with pytest.raises(LLMServiceError):
        asyncio.run(service.generate_response("Hello"))


def test_empty_answer_raises_llm_service_error() -> None:
    service, _ = make_service([make_response(content=None)])

    with pytest.raises(LLMServiceError):
        asyncio.run(service.generate_response("Hello"))


def test_endless_tool_calls_stop_after_max_calls() -> None:
    tool_reply = make_response(
        tool_calls=[make_tool_call("get_employee_leave_balance", '{"employee_id": 101}')]
    )
    service, fake_chat = make_service([tool_reply] * MAX_LLM_CALLS)

    with pytest.raises(LLMServiceError):
        asyncio.run(service.generate_response("Loop forever"))

    assert len(fake_chat.requests) == MAX_LLM_CALLS
