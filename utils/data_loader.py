from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import normalize_counts


DEMO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = DEMO_DIR / "data"
PROFILE_PATH = DEMO_DIR / "agent_profile" / "descriptions" / "wvs_demographic_descriptions_100.json"
ASSET_DIR = DEMO_DIR / "assets"
SIMVBG_DIR = DEMO_DIR / "outputs" / "simvbg"
SIMVBG_METHOD_NAME = "SimVBG"


def output_root() -> Path:
    for candidate in (DEMO_DIR / "outputs", DEMO_DIR / "output"):
        if candidate.exists():
            return candidate
    return DEMO_DIR / "outputs"


def _question_sort_key(qid: str) -> tuple[int, str]:
    match = re.search(r"\d+", qid)
    return (int(match.group(0)) if match else 10_000, qid)


def _numeric_option_sort_key(option: str) -> tuple[float, str]:
    try:
        return (float(option), option)
    except (TypeError, ValueError):
        return (float("inf"), str(option))


@lru_cache(maxsize=1)
def load_questions() -> dict[str, dict[str, Any]]:
    path = DATA_DIR / "questions.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def question_choices() -> list[str]:
    questions = load_questions()
    return [
        f"{qid} - {questions[qid].get('question_text', 'Untitled question')}"
        for qid in sorted(questions, key=_question_sort_key)
    ]


def qid_from_choice(choice: str) -> str:
    return choice.split(" - ", 1)[0].strip()


def question_options(qid: str) -> dict[str, str]:
    raw_options = load_questions().get(qid, {}).get("options", {})
    return {
        str(k): str(v)
        for k, v in sorted(raw_options.items(), key=lambda item: _numeric_option_sort_key(str(item[0])))
    }


def question_text(qid: str) -> str:
    return str(load_questions().get(qid, {}).get("question_text", qid))


@lru_cache(maxsize=1)
def load_wvs() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "WVS_Cross-National_Wave_7_csv_v6_0_100.csv", low_memory=False)


def _coerce_answer(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if not text or text in {"FAILED_TO_PARSE_NUMBER", "nan", "NaN", "None"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    number = float(match.group(0))
    if number.is_integer():
        return str(int(number))
    return str(number)


def human_distribution(qid: str) -> dict[str, Any]:
    options = question_options(qid)
    option_codes = list(options.keys())
    df = load_wvs()
    if qid not in df.columns:
        counts = [0 for _ in option_codes]
        return {"counts": counts, "probs": normalize_counts(counts), "valid_n": 0}

    values = pd.to_numeric(df[qid], errors="coerce").dropna()
    valid_values = [_coerce_answer(value) for value in values]
    counts = [sum(value == code for value in valid_values) for code in option_codes]
    return {"counts": counts, "probs": normalize_counts(counts), "valid_n": int(sum(counts))}


def available_models() -> list[str]:
    root = output_root()
    if not root.exists():
        return []
    return sorted([path.name for path in root.iterdir() if path.is_dir() and path.name != "simvbg"])


def available_methods(model_name: str) -> list[str]:
    model_dir = output_root() / model_name
    if not model_dir.exists():
        return []
    methods = [path.name for path in model_dir.iterdir() if path.is_dir()]
    if simvbg_path_for_model(model_name).exists():
        methods.append(SIMVBG_METHOD_NAME)
    return sorted(methods)


def simvbg_path_for_model(model_name: str) -> Path:
    return SIMVBG_DIR / f"results_{model_name}_100.xlsx"


@lru_cache(maxsize=16)
def load_simvbg_results(model_name: str) -> pd.DataFrame:
    path = simvbg_path_for_model(model_name)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_excel(path)


def simvbg_distribution(model_name: str, qid: str) -> dict[str, Any]:
    options = question_options(qid)
    option_codes = list(options.keys())
    df = load_simvbg_results(model_name)
    if df.empty or qid not in df.columns:
        counts = [0 for _ in option_codes]
        return {
            "counts": counts,
            "probs": normalize_counts(counts),
            "valid_n": 0,
            "invalid_n": int(len(df)) if not df.empty else 0,
            "total_files": int(len(df)),
        }

    values = [_coerce_answer(value) for value in df[qid].tolist()]
    counts = [sum(value == code for value in values) for code in option_codes]
    valid_n = int(sum(counts))
    return {
        "counts": counts,
        "probs": normalize_counts(counts),
        "valid_n": valid_n,
        "invalid_n": int(len(values) - valid_n),
        "total_files": int(len(values)),
    }


def _extract_response(payload: dict[str, Any], qid: str) -> str | None:
    question_payload = payload.get(qid) or payload.get(qid.lower()) or payload.get(qid.upper())
    if isinstance(question_payload, dict):
        for field in ("result", "answer", "response", "value", "output"):
            answer = _coerce_answer(question_payload.get(field))
            if answer is not None:
                return answer
    return _coerce_answer(question_payload)


@lru_cache(maxsize=2048)
def simulation_distribution(model_name: str, method_name: str, qid: str) -> dict[str, Any]:
    if method_name == SIMVBG_METHOD_NAME:
        return simvbg_distribution(model_name, qid)

    options = question_options(qid)
    option_codes = list(options.keys())
    counts = {code: 0 for code in option_codes}
    parsed_n = 0
    invalid_n = 0
    method_dir = output_root() / model_name / method_name

    for file_path in sorted(method_dir.glob("*.json"), key=lambda path: _question_sort_key(path.stem)):
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            invalid_n += 1
            continue
        answer = _extract_response(payload, qid)
        if answer in counts:
            counts[answer] += 1
            parsed_n += 1
        else:
            invalid_n += 1

    count_values = [counts[code] for code in option_codes]
    return {
        "counts": count_values,
        "probs": normalize_counts(count_values),
        "valid_n": parsed_n,
        "invalid_n": invalid_n,
        "total_files": len(list(method_dir.glob("*.json"))),
    }


@lru_cache(maxsize=1)
def sample_profiles(limit: int = 3) -> list[str]:
    if not PROFILE_PATH.exists():
        return []
    with PROFILE_PATH.open("r", encoding="utf-8") as handle:
        profiles = json.load(handle)
    return [profiles[key] for key in sorted(profiles, key=lambda key: int(key))[:limit]]
