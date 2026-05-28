"""Red baselines — rule-based and LLM-driven.

Two reference agents in this file:
  - BaselineRedAgent — deterministic, no API key. Composes four
    transformations on the ground truth (swap digits, perturb value,
    shift validation key, fabricate with existing value).
  - LLMRedAgent     — minimal Groq-driven baseline using gpt-oss-120b.

Pick one in red/submission.py. Both are intentionally weak — students
should outperform them.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from shared.metering import record_llm_usage
from shared.types import GroundTruth, KPI, RedExtraction

from .base import RedAgent


# ─────────────────────── Rule-based baseline ───────────────────────


class BaselineRedAgent(RedAgent):
    """Stochastic rule-based baseline.

    Per call:
      - Pick a random fraction of GT to use as base (70–100%), so red keeps
        well over half the GT and stays inside the coverage quota.
      - Pick random indices to corrupt so ~20% of the output is hallucinated
        (well under the 25%-of-GT addition cap).
      - For each picked index, pick a random transformation among:
        swap_two_digits, perturb_value (×1.01), shift_validation_key.
      - Append one fabricated KPI whose value is borrowed from GT.

    The randomness makes the agent harder to game even though everything
    is rule-based. Pass `seed` to make a run reproducible.
    """

    name = "rule-red"

    _BASE_FRACTION_RANGE = (0.70, 1.0)
    _HALLUC_RATE = 0.20  # target hallucination share of the output

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def extract(
        self, document_text: str, ground_truth: GroundTruth
    ) -> RedExtraction:
        gt = list(ground_truth.kpis)
        if not gt:
            return RedExtraction(kpis=[])

        # Random base size.
        frac = self._rng.uniform(*self._BASE_FRACTION_RANGE)
        n_base = min(max(3, int(len(gt) * frac)), len(gt))

        kpis: list[KPI] = [self._copy(i, k) for i, k in enumerate(gt[:n_base])]

        # Target ~20% hallucination rate on (n_base + 1 fabricated).
        target_halluc = max(1, int(round((n_base + 1) * self._HALLUC_RATE)))
        n_modify = max(0, min(target_halluc - 1, n_base))

        indices = self._rng.sample(range(n_base), n_modify) if n_modify else []
        for idx in indices:
            self._try_random_modification(kpis, idx, ground_truth)

        fab = self.fabricate_with_existing_value(
            ground_truth, len(kpis), used_names={k.name for k in kpis}
        )
        if fab is not None:
            kpis.append(fab)

        return RedExtraction(kpis=kpis)

    def _try_random_modification(
        self,
        kpis: list[KPI],
        idx: int,
        ground_truth: GroundTruth,
    ) -> None:
        options = [
            self.swap_two_digits,
            lambda k: self.perturb_value(k, 1.01),
            lambda k: self.shift_validation_key(k, ground_truth),
        ]
        self._rng.shuffle(options)
        for fn in options:
            modified = fn(kpis[idx])
            if modified is not None and modified != kpis[idx]:
                kpis[idx] = modified
                return

    @staticmethod
    def _copy(new_id: int, k: KPI) -> KPI:
        return KPI(
            id=new_id,
            name=k.name,
            value=k.value,
            unit=k.unit,
            period=k.period,
            scope=k.scope,
            source_span=k.source_span,
        )

    @staticmethod
    def swap_two_digits(kpi: KPI) -> KPI | None:
        """Swap two adjacent digits in the numeric value (e.g. 14876 → 14867)."""
        if not isinstance(kpi.value, (int, float)) or kpi.value == 0:
            return None
        chars = list(repr(kpi.value))
        for i in range(len(chars) - 1, 0, -1):
            if (
                chars[i].isdigit()
                and chars[i - 1].isdigit()
                and chars[i] != chars[i - 1]
            ):
                chars[i], chars[i - 1] = chars[i - 1], chars[i]
                try:
                    new_val = float("".join(chars))
                except ValueError:
                    return None
                if new_val != kpi.value:
                    return kpi.model_copy(update={"value": new_val})
                return None
        return None

    @staticmethod
    def perturb_value(kpi: KPI, factor: float = 1.01) -> KPI | None:
        """Multiply the numeric value by `factor` (default +1%)."""
        if not isinstance(kpi.value, (int, float)):
            return None
        new_val = kpi.value * factor
        if new_val == kpi.value:
            return None
        return kpi.model_copy(update={"value": new_val})

    @staticmethod
    def shift_validation_key(kpi: KPI, ground_truth: GroundTruth) -> KPI | None:
        """Shift the period to a nearby year that doesn't exist in GT."""
        if not kpi.period:
            return None
        try:
            year = int(kpi.period)
        except (ValueError, TypeError):
            return None
        gt_keys = {(k.name, k.period, k.scope) for k in ground_truth.kpis}
        for delta in (-1, 1, -2, 2):
            new_period = str(year + delta)
            if (kpi.name, new_period, kpi.scope) not in gt_keys:
                return kpi.model_copy(update={"period": new_period})
        return None

    @staticmethod
    def fabricate_with_existing_value(
        ground_truth: GroundTruth, new_id: int, used_names: set[str]
    ) -> KPI | None:
        """New KPI whose value is borrowed from GT (so it's in the document)
        but whose name is a slight variation not present in GT."""
        gt_keys = {(k.name, k.period, k.scope) for k in ground_truth.kpis}
        for donor in ground_truth.kpis:
            if not isinstance(donor.value, (int, float)):
                continue
            new_name = f"{donor.name} (adjusted)"
            if (new_name, donor.period, donor.scope) in gt_keys:
                continue
            if new_name in used_names:
                continue
            return KPI(
                id=new_id,
                name=new_name,
                value=donor.value,
                unit=donor.unit,
                period=donor.period,
                scope=donor.scope,
            )
        return None


# ─────────────────────── LLM baseline ───────────────────────


class LLMRedAgent(RedAgent):
    """Minimal Groq-driven red agent using gpt-oss-120b.

    Framed as building an adversarial QA dataset (not lying about data) so
    the model doesn't refuse. Single LLM call returns the full
    RedExtraction. Empty extraction on parse error — quota fails downstream
    as a clear signal to the student.

    Setup: `pip install -e ".[llm]"`. For the tournament, set LLM_BASE_URL +
    LLM_API_KEY (your team's proxy virtual key) in template/.env; for local dev
    straight against Groq, set GROQ_API_KEY instead.
    """

    name = "llm-red"

    _DEFAULT_MODEL = "openai/gpt-oss-120b"
    _BASE_URL = "https://api.groq.com/openai/v1"
    _GT_CAP = 20

    _INSTRUCTIONS = """You are building an adversarial quality-assurance dataset
to test a downstream KPI verifier.

You are given a source document and a list of ground-truth KPIs extracted
from it. Produce an adversarial version of that list. Between 15% and 25%
of the KPIs in your output must be intentionally wrong, in one of three ways:
  - change the numeric value of a real KPI (small deltas stealth better);
  - change the unit, scope, or period of a real KPI;
  - add a brand-new KPI that does not exist in the document.

For every other KPI, copy it verbatim from the ground truth. Do NOT flag
which KPIs you tampered with — the grader derives that from the ground truth.

Reply with one JSON object — no prose, no markdown fence:
{
  "kpis": [
    {"id": 0, "name": "...", "value": ..., "unit": "...", "period": "...", "scope": "..."},
    ...
  ]
}

Use sequential integer ids starting at 0."""

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        from openai import OpenAI  # local import so the rule-based baseline stays dep-free

        self._load_dotenv()
        # Set LLM_BASE_URL + LLM_API_KEY (your team's proxy virtual key) to route
        # through the metering proxy; falls back to Groq directly for local dev.
        base_url = os.environ.get("LLM_BASE_URL", self._BASE_URL)
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key. Set LLM_API_KEY (proxy virtual key) or GROQ_API_KEY "
                "in template/.env or the environment."
            )
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def extract(
        self, document_text: str, ground_truth: GroundTruth
    ) -> RedExtraction:
        gt_payload = [
            {
                "name": k.name,
                "value": k.value,
                "unit": k.unit,
                "period": k.period,
                "scope": k.scope,
            }
            for k in ground_truth.kpis[: self._GT_CAP]
        ]
        user_input = (
            "DOCUMENT:\n"
            f"{document_text}\n\n"
            "GROUND-TRUTH KPIs (clean reference — keep >=50% faithful, "
            "add <=25% of this count as hallucinations):\n"
            f"{json.dumps(gt_payload, ensure_ascii=False, indent=2)}"
        )

        response = self._client.responses.create(
            model=self._model,
            instructions=self._INSTRUCTIONS,
            input=user_input,
        )
        _record_usage(response)
        raw = (response.output_text or "").strip()
        return self._parse_response(raw)

    @staticmethod
    def _load_dotenv() -> None:
        if os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY"):
            return
        here = Path(__file__).resolve()
        for candidate in (
            here.parent.parent / ".env",
            here.parent.parent.parent / ".env",
        ):
            if not candidate.exists():
                continue
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            if os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY"):
                return

    @staticmethod
    def _parse_response(raw: str) -> RedExtraction:
        # Strip markdown code fences if the model wrapped its JSON.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return RedExtraction(kpis=[])

        kpis: list[KPI] = []
        for raw_kpi in payload.get("kpis") or []:
            try:
                kpis.append(KPI(**raw_kpi))
            except (TypeError, ValueError):
                continue

        return RedExtraction(kpis=kpis)


def _record_usage(response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    record_llm_usage(
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )
