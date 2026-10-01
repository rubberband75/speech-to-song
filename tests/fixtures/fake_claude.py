"""A stand-in for the Anthropic client: records requests, returns scripted responses."""

import json
from types import SimpleNamespace
from typing import Any


def usage_item(kind: str, model: str, input_tokens: int, output_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        type=kind,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def response(
    data: Any,
    *,
    model: str = "claude-sonnet-5-5",
    stop_reason: str = "end_turn",
    input_tokens: int = 1000,
    output_tokens: int = 500,
    iterations: list[SimpleNamespace] | None = None,
    category: str | None = None,
) -> SimpleNamespace:
    """A Messages API response carrying `data` as the JSON text block."""
    text = data if isinstance(data, str) else json.dumps(data)
    content = [] if stop_reason == "refusal" else [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text=text),
    ]  # fmt: skip
    return SimpleNamespace(
        id="msg_fake",
        _request_id="req_fake",
        model=model,
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(type="refusal", category=category) if category else None,
        content=content,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            iterations=iterations,
        ),
    )


class FakeAnthropic:
    """`client.beta.messages.create(**kwargs)` pops the next scripted response."""

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected Claude call: no scripted response left")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
