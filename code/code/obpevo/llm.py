"""LLM back-ends and the logging wrapper every search method talks to.

BLADE principle: cost accounting happens *outside* the search method. Every method gets a
`LoggedLLM`, which records each call (tokens, latency, cost) to llm_calls.csv and enforces
the shared token / wall-clock budget, whatever the method does internally.
"""

from __future__ import annotations

import csv
import re
import os
import threading
import time
from pathlib import Path

import numpy as np
from llm4ad.base import LLM


class BudgetExhausted(KeyboardInterrupt):
    """Raised when a shared budget runs out. Subclasses KeyboardInterrupt on purpose: LLM4AD
    methods catch `Exception` and keep looping, but stop cleanly on KeyboardInterrupt.
    """


# ----------------------------------------------------------------------------- back-ends
class _HTTPClient:
    """Shared request/retry logic. Retries rate limits, server errors and network errors with exponential
    back-off; fails immediately on other 4xx (bad key, unknown model) and shows the provider's message.
    """

    def __init__(self, timeout: int, max_retries: int):
        import requests

        self._session, self.timeout, self.max_retries = (
            requests.Session(),
            timeout,
            max_retries,
        )

    def _post(self, url: str, body: dict, headers: dict) -> dict:
        import requests

        for attempt in range(self.max_retries):
            try:
                r = self._session.post(
                    url, json=body, headers=headers, timeout=self.timeout
                )
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    raise
            else:
                if r.ok:
                    return r.json()
                if r.status_code != 429 and r.status_code < 500:
                    raise RuntimeError(
                        f"HTTP {r.status_code} from {url}: {r.text[:500]}"
                    )
                if attempt == self.max_retries - 1:
                    r.raise_for_status()
            time.sleep(2**attempt)


def _api_key(env: str | None, required: bool = True) -> str:
    key = os.environ.get(env, "") if env else ""
    if required and not key:
        raise RuntimeError(f"Set the {env} environment variable (see readme.md)")
    return key


class OpenAICompatLLM(_HTTPClient):
    """Any OpenAI-compatible chat API: OpenAI, Gemini, DeepSeek, OpenRouter, Ollama, vLLM.
    `extra` is merged into the request body for provider-specific options, e.g. {reasoning_effort: low}.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str | None = None,
        temperature: float | None = 1.0,
        max_tokens: int = 4096,
        token_param: str = "max_tokens",
        extra: dict | None = None,
        timeout: int = 180,
        max_retries: int = 5,
        **_,
    ):
        super().__init__(timeout, max_retries)
        self.model, self.url = model, base_url.rstrip("/") + "/chat/completions"
        key = _api_key(api_key_env, required=api_key_env is not None)
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.body = {"model": model, token_param: max_tokens, **(extra or {})}
        if temperature is not None:  # some reasoning models only accept the default
            self.body["temperature"] = temperature

    def complete(self, prompt: str):
        data = self._post(
            self.url,
            {**self.body, "messages": [{"role": "user", "content": prompt}]},
            self.headers,
        )
        text = data["choices"][0]["message"].get("content") or ""
        u = data.get("usage") or None
        if (
            u
        ):  # count hidden reasoning tokens too: some providers leave them out of completion_tokens
            p = u.get("prompt_tokens") or 0
            u = {
                "prompt_tokens": p,
                "completion_tokens": max(
                    u.get("completion_tokens") or 0, (u.get("total_tokens") or 0) - p
                ),
            }
        return text, u


class AnthropicLLM(_HTTPClient):
    """Anthropic's native Messages API (Claude models)."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.anthropic.com/v1",
        api_key_env: str = "ANTHROPIC_API_KEY",
        temperature: float | None = 1.0,
        max_tokens: int = 4096,
        extra: dict | None = None,
        timeout: int = 180,
        max_retries: int = 5,
        **_,
    ):
        super().__init__(timeout, max_retries)
        self.url = base_url.rstrip("/") + "/messages"
        self.headers = {
            "x-api-key": _api_key(api_key_env),
            "anthropic-version": "2023-06-01",
        }
        self.body = {"model": model, "max_tokens": max_tokens, **(extra or {})}
        if temperature is not None:
            self.body["temperature"] = min(temperature, 1.0)  # Anthropic accepts 0-1

    def complete(self, prompt: str):
        data = self._post(
            self.url,
            {**self.body, "messages": [{"role": "user", "content": prompt}]},
            self.headers,
        )
        text = "".join(
            b.get("text", "")
            for b in data.get("content", [])
            if b.get("type") == "text"
        )
        u = data.get("usage") or {}
        return text, {
            "prompt_tokens": u.get("input_tokens", 0),
            "completion_tokens": u.get("output_tokens", 0),
        }


class FakeLLM:
    """Offline stand-in for smoke tests: returns plausible heuristics with random constants,
    plus occasional broken ones (NaN, wrong shape, infinite loop) to exercise failure logging.
    Responses follow EoH's format ({thought} + code), which every LLM4AD method can parse.
    """

    BODIES = [  # (weight, thought, body); {a}, {b}, {t} are filled with random constants
        (
            3,
            "Best fit: prefer the bin with the least space left",
            "    return item - bins",
        ),
        (
            3,
            "Best fit, but penalise leaving a small awkward gap of under {a} units",
            "    r = bins - item\n    s = -r.astype(float)\n    s[(r > 0) & (r < {a})] -= {b}\n    return s",
        ),
        (
            2,
            "Exponential preference for tight fits with an exact-fit bonus",
            "    r = bins - item\n    return np.exp(-r / {t}) + {b} * (r == 0)",
        ),
        (
            2,
            "Prefer bins that leave close to a target leftover of {a} units",
            "    r = bins - item\n    return -np.abs(r - {a}) * (r > 0) - r * 0.01",
        ),
        (
            1,
            "FunSearch-style combination of capacity powers",
            "    m = bins.max()\n    s = (bins - m) ** 2 / item + bins ** 2 / item ** 2 + bins ** 2 / item ** 3\n"
            "    s[bins > item] = -s[bins > item]\n    s[1:] -= s[:-1]\n    return s",
        ),
        (1, "First fit: earliest feasible bin", "    return -np.arange(len(bins))"),
        (1, "Worst fit: prefer the emptiest bin", "    return bins - item"),
        (
            1,
            "Square-root slack score",
            "    return np.sqrt(item - bins)",
        ),  # NaN -> error status
        (
            1,
            "Drop the last bin",
            "    return (item - bins)[:-1]",
        ),  # wrong shape -> error
        (0.3, "Think forever", "    while True:\n        pass"),  # -> timeout status
    ]

    TARGET = 3  # index of the target-leftover family in BODIES

    def __init__(self, seed: int = 0, **_):
        self.rng, self.lock = np.random.default_rng(seed), threading.Lock()
        w = np.array([b[0] for b in self.BODIES], float)
        self.p = w / w.sum()

    def complete(self, prompt: str):
        # Crude stand-in for "learning from parents": if the prompt shows a parent from the target-leftover
        # family, sometimes nudge its constant. Methods that put parents in prompts (EoH, ReEvo, HillClimb,
        # FunSearch) can then improve on random sampling, which checks parents really flow through prompts.
        parents = re.findall(r"np\.abs\(r - (\d+)\)", prompt)
        with self.lock:
            k = self.rng.choice(len(self.BODIES), p=self.p)
            consts = dict(
                a=int(self.rng.integers(2, 15)),
                b=round(float(self.rng.uniform(0.5, 20)), 2),
                t=round(float(self.rng.uniform(1, 10)), 2),
            )
            if parents and self.rng.random() < 0.6:
                k = self.TARGET
                consts["a"] = max(
                    1, int(self.rng.choice(parents)) + int(self.rng.choice([-1, 1]))
                )
        _, thought, body = self.BODIES[k]
        thought, body = thought.format(**consts), body.format(**consts)
        hint = f"Hint: prefer tight fits but avoid leaving gaps smaller than {consts['a']} units."  # used by ReEvo reflections
        return (
            f"{{{thought}}}\n{hint}\n\n```python\nimport numpy as np\n\ndef priority(item, bins):\n{body}\n```\n",
            None,
        )


# Provider presets: a config only needs `provider` and `model`; any field here can still be overridden in the config.
PROVIDERS = {
    "openai": dict(
        cls="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        token_param="max_completion_tokens",
    ),
    "gemini": dict(
        cls="openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY",
    ),
    "anthropic": dict(
        cls="anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "deepseek": dict(
        cls="openai",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    "openrouter": dict(
        cls="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    ),
    "ollama": dict(
        cls="openai", base_url="http://localhost:11434/v1", api_key_env=None
    ),  # local, no key
    "openai_compat": dict(
        cls="openai"
    ),  # any other OpenAI-style server: give base_url, api_key_env
    "fake": dict(cls="fake"),
}
_CLASSES = {"openai": OpenAICompatLLM, "anthropic": AnthropicLLM, "fake": FakeLLM}


def make_backend(llm_cfg: dict, seed: int = 0):
    """Build the client for a config's `llm` section: preset for the provider, then the config's own fields on top."""
    provider = llm_cfg.get("provider") or llm_cfg.get(
        "backend", "fake"
    )  # `backend` kept for old configs
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}; choose from {sorted(PROVIDERS)}"
        )
    opts = {
        **PROVIDERS[provider],
        **{k: v for k, v in llm_cfg.items() if v is not None or k == "temperature"},
    }
    return _CLASSES[opts.pop("cls")](seed=seed, **opts)


# ----------------------------------------------------------------------------- logging wrapper
class LoggedLLM(LLM):
    """LLM4AD-compatible LLM that logs every call and enforces the shared budget."""

    FIELDS = [
        "call",
        "t_start",
        "latency_s",
        "prompt_tokens",
        "completion_tokens",
        "token_source",
        "cum_tokens",
        "cost_usd",
        "status",
        "prompt_chars",
        "response_chars",
    ]

    def __init__(
        self,
        backend,
        log_path: Path,
        max_tokens: int | None = None,
        max_minutes: float | None = None,
        price_in_per_1m: float = 0.0,
        price_out_per_1m: float = 0.0,
        prompt_suffix: str = "",
    ):
        super().__init__(do_auto_trim=True)
        self.prompt_suffix = (
            prompt_suffix  # e.g. task.FORMAT_HINT, added to every prompt
        )
        self.backend, self.log_path = backend, Path(log_path)
        self.max_tokens, self.deadline = max_tokens, (
            (time.time() + 60 * max_minutes) if max_minutes else None
        )
        self.price_in, self.price_out = price_in_per_1m / 1e6, price_out_per_1m / 1e6
        self.lock, self.calls, self.tokens, self.cost, self.exhausted = (
            threading.Lock(),
            0,
            0,
            0.0,
            None,
        )
        self.fail_streak, self.max_fail_streak = (
            0,
            5,
        )  # stop the run if the API keeps failing (e.g. daily quota)
        # Set to the EvalRecorder by the runner: once the evaluation budget is spent, refuse further calls too.
        # Without this, an LLM4AD sampler thread whose evaluation pool was shut down by another thread
        # keeps calling the LLM (and spending money) without ever evaluating anything.
        self.linked_budget = None
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)

    def _check_budget(self):
        if self.linked_budget is not None and self.linked_budget.exhausted:
            self.exhausted = self.linked_budget.exhausted
        elif self.max_tokens is not None and self.tokens >= self.max_tokens:
            self.exhausted = "tokens"
        elif self.deadline is not None and time.time() > self.deadline:
            self.exhausted = "wall_clock"
        if self.exhausted:
            raise BudgetExhausted(self.exhausted)

    def draw_sample(self, prompt, *args, **kwargs) -> str:
        prompt = prompt if isinstance(prompt, str) else str(prompt)
        if self.prompt_suffix:
            prompt = f"{prompt.rstrip()}\n\n{self.prompt_suffix}"
        with self.lock:
            self._check_budget()
            call = self.calls = self.calls + 1
        t0 = time.time()
        try:
            text, usage, status = *self.backend.complete(prompt), "ok"
        except Exception as e:  # API failure after retries
            text, usage, status = "", None, f"error:{type(e).__name__}: {str(e)[:150]}"
        if status == "ok":  # ~4 chars per token if the API gives no usage
            p_tok = (usage or {}).get("prompt_tokens") or len(prompt) // 4
            c_tok = (usage or {}).get("completion_tokens") or len(text) // 4
        else:  # failed calls aren't billed: count nothing
            p_tok = c_tok = 0
        cost = p_tok * self.price_in + c_tok * self.price_out
        with self.lock:
            self.tokens += p_tok + c_tok
            self.cost += cost
            row = [
                call,
                round(t0, 3),
                round(time.time() - t0, 3),
                p_tok,
                c_tok,
                "api" if usage else "estimate",
                self.tokens,
                round(cost, 6),
                status,
                len(prompt),
                len(text),
            ]
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        with self.lock:
            self.fail_streak = 0 if status == "ok" else self.fail_streak + 1
            if self.fail_streak >= self.max_fail_streak and not self.exhausted:
                self.exhausted = (
                    "llm_errors"  # ends the run like a budget; results so far are kept
                )
        if self.exhausted == "llm_errors":
            raise BudgetExhausted("llm_errors")
        if status != "ok":
            raise RuntimeError(f"LLM call failed: {status}")
        return text
