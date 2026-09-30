import json
import logging
import os
from typing import Any

from openrouter import OpenRouter
from pydantic import BaseModel, ValidationError

from enterprise_ai_knowledge_assistant.services.employee_service import (
    employee_service,
)

logger = logging.getLogger(__name__)

MODEL = "openrouter/free"

# Safety net: stop if the LLM keeps requesting tools and never gives an answer.
MAX_LLM_CALLS = 5

SYSTEM_PROMPT = """
You are an enterprise knowledge assistant.

Your responsibilities:
- Answer questions clearly and accurately.
- Use the provided enterprise knowledge when available.
- When a question needs employee data (such as leave balances), use the available tools instead of guessing.
- Do not invent company policies or facts.
- If the required information is unavailable, clearly say that you do not have enough information.
- Keep answers concise and professional.
"""

# 1. What the LLM SEES: a JSON Schema description of each tool.
#    The LLM never sees our Python code, only this description.
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_employee_leave_balance",
            "description": (
                "Get an employee's name and number of remaining leave days, "
                "looked up by their numeric employee ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {
                        "type": "integer",
                        "description": "The numeric employee ID, for example 101.",
                    },
                },
                "required": ["employee_id"],
                "additionalProperties": False,
            },
        },
    },
]


# 2. How the BACKEND validates the arguments the LLM sends back.
class EmployeeLeaveBalanceArgs(BaseModel):
    employee_id: int


# 3. What the BACKEND RUNS: tool name -> (argument model, Python function).
#    Only functions listed here can ever be executed.
AVAILABLE_TOOLS = {
    "get_employee_leave_balance": (
        EmployeeLeaveBalanceArgs,
        employee_service.get_employee_leave_balance,
    ),
}


class LLMServiceError(Exception):
    """Raised when the LLM call fails or returns something unusable."""


def execute_tool(name: str, arguments_json: str) -> dict[str, Any]:
    """Run one tool requested by the LLM and return its result.

    Problems are returned as {"error": ...} instead of raised, so the LLM
    can read them and explain the problem to the user.
    """
    if name not in AVAILABLE_TOOLS:
        logger.warning("LLM requested unknown tool %r", name)
        return {"error": f"Unknown tool: {name}"}

    args_model, function = AVAILABLE_TOOLS[name]

    try:
        # Parses the JSON string AND checks the types in one step.
        args = args_model.model_validate_json(arguments_json)
    except ValidationError as exc:
        logger.warning("Invalid arguments for tool %r: %r", name, arguments_json)
        problems = "; ".join(error["msg"] for error in exc.errors())
        return {"error": f"Invalid arguments for {name}: {problems}"}

    try:
        result = function(**args.model_dump())
    except Exception:
        logger.exception("Tool %r failed", name)
        return {"error": f"Tool {name} failed to run"}

    logger.info("Tool %s(%s) returned %s", name, args.model_dump(), result)
    return result


class LLMService:
    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is not set"
            )

        self.client = OpenRouter(
            api_key=api_key
        )

    async def generate_response(self, prompt: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        for call_number in range(1, MAX_LLM_CALLS + 1):
            logger.info("LLM call %d: sending %d messages", call_number, len(messages))
            message = await self._call_llm(messages)

            tool_calls = message.tool_calls or []

            # No tool requested: this is the final answer.
            if not tool_calls:
                logger.info("LLM call %d: final answer received", call_number)
                return self._get_answer_text(message)

            logger.info(
                "LLM call %d: LLM requested tools %s",
                call_number,
                [tool_call.function.name for tool_call in tool_calls],
            )

            # Record the assistant's tool request in the conversation.
            # The follow-up call needs it, or the tool results below would
            # answer a question the LLM never asked.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content if isinstance(message.content, str) else None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in tool_calls
                    ],
                }
            )

            # The backend (not the LLM) runs each tool, then adds its result.
            # tool_call_id links each result to the request it answers.
            for tool_call in tool_calls:
                result = execute_tool(tool_call.function.name, tool_call.function.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

            # Loop again: the LLM now reads the tool results and writes the answer.

        raise LLMServiceError(
            f"The language model did not produce an answer after {MAX_LLM_CALLS} calls"
        )

    async def _call_llm(self, messages: list[dict[str, Any]]) -> Any:
        """Send the conversation plus tool definitions; return the assistant message."""
        try:
            response = await self.client.chat.send_async(
                model=MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",  # the LLM decides whether a tool is needed
            )
        except Exception as exc:
            # Log only the error, never the request (it would carry the API key header).
            logger.error("OpenRouter request failed: %s", exc)
            raise LLMServiceError("The language model request failed") from exc

        if not response or not response.choices:
            raise LLMServiceError("The language model returned no choices")

        return response.choices[0].message

    @staticmethod
    def _get_answer_text(message: Any) -> str:
        if not isinstance(message.content, str) or not message.content.strip():
            raise LLMServiceError("The language model returned an empty answer")
        return message.content
