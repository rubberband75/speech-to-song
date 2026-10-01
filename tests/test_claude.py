"""The Claude wrapper, with a fake client (never the network)."""

from pathlib import Path

import pytest

from speech2song.config import AppConfig
from speech2song.costs import CostLog
from speech2song.errors import S2SError
from speech2song.llm import claude
from speech2song.llm.claude import FALLBACK_BETA, ClaudeJson

from .fixtures.fake_claude import FakeAnthropic, response, usage_item

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


@pytest.fixture
def log(tmp_path: Path) -> CostLog:
    return CostLog(tmp_path / "costs.json")


def _client(
    log: CostLog, fake: FakeAnthropic, monkeypatch: pytest.MonkeyPatch, **config: object
) -> ClaudeJson:
    monkeypatch.setattr(claude, "client_factory", lambda: fake)
    return ClaudeJson(AppConfig(**config), log, "run-1", "claude-sonnet-5-5")


def _call(client: ClaudeJson):
    return client.call(system="sys", user="hi", schema=SCHEMA, stage="select", operation="op")


def test_request_shape_and_cost_entry(log: CostLog, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAnthropic(response({"ok": True}, input_tokens=1000, output_tokens=500))
    reply = _call(_client(log, fake, monkeypatch))
    assert reply.data == {"ok": True}
    request = fake.requests[0]
    assert request["model"] == "claude-sonnet-5-5"
    assert request["system"] == "sys"
    assert request["messages"] == [{"role": "user", "content": "hi"}]
    assert request["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert (request["fallbacks"], request["betas"]) == ("default", [FALLBACK_BETA])
    [entry] = log.read()
    assert (entry.service, entry.stage, entry.model, entry.estimated) == (
        "anthropic",
        "select",
        "claude-sonnet-5-5",
        False,
    )
    assert entry.units["input_tokens"] == 1000 and entry.units["output_tokens"] == 500
    assert entry.usd == pytest.approx((1000 * 2 + 500 * 10) / 1e6)
    assert entry.request_id == "req_fake"
    assert reply.usd == entry.usd


def test_fallbacks_can_be_turned_off(log: CostLog, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAnthropic(response({}))
    _call(_client(log, fake, monkeypatch, claude_fallbacks=False))
    assert "fallbacks" not in fake.requests[0] and "betas" not in fake.requests[0]


def test_fallback_attempts_are_priced_by_the_model_that_ran(
    log: CostLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    iterations = [
        usage_item("message", "claude-sonnet-5-5", 1000, 0),
        usage_item("fallback_message", "claude-opus-4-8", 1200, 600),
    ]
    fake = FakeAnthropic(response({}, model="claude-opus-4-8", iterations=iterations))
    reply = _call(_client(log, fake, monkeypatch))
    [entry] = log.read()
    assert entry.model == "claude-opus-4-8"
    assert entry.usd == pytest.approx((1000 * 2 + 1200 * 5 + 600 * 25) / 1e6)
    assert entry.note == "claude-sonnet-5-5 declined; claude-opus-4-8 answered"
    assert reply.fallback_from == "claude-sonnet-5-5"


def test_unpriced_model_is_logged_as_estimated(
    log: CostLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeAnthropic(response({}, model="claude-mystery"))
    reply = _call(_client(log, fake, monkeypatch))
    [entry] = log.read()
    assert entry.estimated and "no price configured" in (entry.note or "")
    assert reply.usd is None


def test_refusal_is_an_error_but_still_logged(
    log: CostLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeAnthropic(response({}, stop_reason="refusal", category="general_harms"))
    with pytest.raises(S2SError, match=r"declined.*general_harms"):
        _call(_client(log, fake, monkeypatch))
    assert len(log.read()) == 1


def test_truncated_answer_is_an_error(log: CostLog, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAnthropic(response('{"clips": [', stop_reason="max_tokens"))
    with pytest.raises(S2SError, match="max_tokens"):
        _call(_client(log, fake, monkeypatch))


def test_invalid_json_is_an_error(log: CostLog, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAnthropic(response("not json"))
    with pytest.raises(S2SError, match="invalid JSON"):
        _call(_client(log, fake, monkeypatch))


def test_api_errors_become_clean_messages(log: CostLog, monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    denied = anthropic.AuthenticationError(
        "invalid x-api-key", response=httpx2.Response(401, request=request), body=None
    )
    with pytest.raises(S2SError, match="rejected the credentials"):
        _call(_client(log, FakeAnthropic(denied), monkeypatch))
    missing = TypeError("Could not resolve authentication method")
    with pytest.raises(S2SError, match="No Anthropic credentials"):
        _call(_client(log, FakeAnthropic(missing), monkeypatch))
    assert log.read() == []
