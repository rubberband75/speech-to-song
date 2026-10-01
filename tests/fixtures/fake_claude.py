"""A stand-in for the Anthropic client: records requests, returns scripted responses."""

import json
from types import SimpleNamespace
from typing import Any

from speech2song.models import Transcript

PICKS = [(1, 1), (2, 3), (5, 5), (7, 7), (8, 8)]


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


def clip_answer(transcript: Transcript, picks: list[tuple[int, int]] = PICKS) -> dict[str, Any]:
    """A valid selection answer for `transcript`: clips c1..cN, played in reverse."""
    clips = [
        {
            "id": f"c{i + 1}",
            "start_sentence": a,
            "end_sentence": b,
            "text": " ".join(s.text for s in transcript.sentences[a - 1 : b]),
            "score": round(0.9 - i * 0.1, 2),
            "role": "hook",
            "reason": "it stands alone",
        }
        for i, (a, b) in enumerate(picks)
    ]
    return {"clips": clips, "suggested_order": [c["id"] for c in reversed(clips)], "notes": "n"}
