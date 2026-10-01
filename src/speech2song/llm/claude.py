"""Thin Claude wrapper: one structured-output (JSON schema) call, logged to costs.json.

Uses the official SDK. Server-side refusal fallbacks (`fallbacks: "default"`) are on by
default: if Claude's safety classifiers decline, the API re-runs the request on a
recommended model. Every attempt is priced at the rates of the model that ran it.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from speech2song.config import AppConfig
from speech2song.costs import CostLog, claude_usd
from speech2song.errors import S2SError
from speech2song.models import CostEntry

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _default_client() -> Any:
    import anthropic

    return anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY (.env is loaded beforehand)


# Tests replace this to inject a fake client; nothing in tests may reach the network.
client_factory: Callable[[], Any] = _default_client


@dataclass
class JsonReply:
    data: Any  # the parsed JSON
    model: str  # the model that produced it
    request_id: str | None
    input_tokens: int
    output_tokens: int
    usd: float | None
    fallback_from: str | None  # requested model, when a fallback model answered


@dataclass
class _Attempt:
    model: str
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_write: int


def _attempts(response: Any, requested: str) -> list[_Attempt]:
    """Per-attempt usage: `usage.iterations` when a fallback ran, else top-level usage."""
    usage = response.usage
    iterations = getattr(usage, "iterations", None) or []
    attempts = [
        _Attempt(
            model=getattr(it, "model", None) or requested,
            input_tokens=it.input_tokens,
            output_tokens=it.output_tokens,
            cache_read=it.cache_read_input_tokens or 0,
            cache_write=it.cache_creation_input_tokens or 0,
        )
        for it in iterations
        if getattr(it, "type", None) in ("message", "fallback_message")
    ]
    if attempts:
        return attempts
    return [
        _Attempt(
            model=response.model or requested,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read=getattr(usage, "cache_read_input_tokens", None) or 0,
            cache_write=getattr(usage, "cache_creation_input_tokens", None) or 0,
        )
    ]


class ClaudeJson:
    def __init__(self, config: AppConfig, cost_log: CostLog, run_id: str, model: str) -> None:
        self.config = config
        self.cost_log = cost_log
        self.run_id = run_id
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = client_factory()
        return self._client

    def request(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        """The Messages API request body (also used to estimate size in dry runs)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.config.claude_max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "effort": self.config.claude_effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        }
        if self.config.claude_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def call(
        self, *, system: str, user: str, schema: dict[str, Any], stage: str, operation: str
    ) -> JsonReply:
        import anthropic

        try:
            response = self._get_client().beta.messages.create(**self.request(system, user, schema))
        except anthropic.AuthenticationError as exc:
            raise S2SError("The Anthropic API rejected the credentials (check .env).") from exc
        except anthropic.NotFoundError as exc:
            raise S2SError(f"Unknown Claude model {self.model!r}: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise S2SError("Claude rate limit reached; try again shortly.") from exc
        except anthropic.APIStatusError as exc:
            raise S2SError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise S2SError(f"Could not reach the Claude API: {exc}") from exc
        except TypeError as exc:  # raised by the SDK when no credentials can be resolved
            if "auth" in str(exc).lower():
                raise S2SError(
                    "No Anthropic credentials: set ANTHROPIC_API_KEY in .env "
                    "(or log in with `ant auth login`)."
                ) from exc
            raise

        reply = self._log(response, stage, operation)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise S2SError(f"Claude declined the request (category: {category}).")
        if response.stop_reason == "max_tokens":
            raise S2SError(
                f"Claude's answer was cut off at max_tokens={self.config.claude_max_tokens}; "
                "raise claude_max_tokens in config.yaml."
            )
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise S2SError("Claude returned no text block.")
        try:
            reply.data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise S2SError(f"Claude returned invalid JSON: {exc}") from exc
        return reply

    def _log(self, response: Any, stage: str, operation: str) -> JsonReply:
        """Append one costs.json entry per call (sums over fallback attempts)."""
        attempts = _attempts(response, self.model)
        pricing = self.config.pricing
        total: float | None = 0.0
        unpriced = []
        for a in attempts:
            usd = claude_usd(
                pricing,
                a.model,
                a.input_tokens,
                a.output_tokens,
                cache_read_tokens=a.cache_read,
                cache_write_tokens=a.cache_write,
            )
            if usd is None:
                unpriced.append(a.model)
            else:
                total = (total or 0.0) + usd
        served = response.model or self.model
        fallback_from = self.model if served != self.model else None
        notes = []
        if fallback_from:
            notes.append(f"{self.model} declined; {served} answered")
        if unpriced:
            notes.append(f"no price configured for {', '.join(sorted(set(unpriced)))}")
        price = pricing.anthropic.get(served)
        input_tokens = sum(a.input_tokens for a in attempts)
        output_tokens = sum(a.output_tokens for a in attempts)
        self.cost_log.append(
            CostEntry(
                ts=datetime.now().astimezone(),
                run_id=self.run_id,
                stage=stage,
                service="anthropic",
                operation=operation,
                model=served,
                units={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_input_tokens": sum(a.cache_read for a in attempts),
                    "cache_creation_input_tokens": sum(a.cache_write for a in attempts),
                },
                usd=round(total or 0.0, 6),
                estimated=bool(unpriced),
                price_ref=f"pricing.anthropic.{served} (as of {price.as_of})" if price else None,
                request_id=getattr(response, "_request_id", None) or getattr(response, "id", None),
                note="; ".join(notes) or None,
            )
        )
        log.info("Claude %s: %d in / %d out tokens, $%.4f", served, input_tokens, output_tokens,
                 total or 0.0)  # fmt: skip
        return JsonReply(
            data=None,
            model=served,
            request_id=getattr(response, "_request_id", None) or getattr(response, "id", None),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=None if unpriced else round(total or 0.0, 6),
            fallback_from=fallback_from,
        )
