from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Optional, Set


@dataclass(frozen=True)
class ChoiceParseResult:
    choice: Optional[str]
    strategy: Optional[str]


def _normalize_response_text(response: str) -> str:
    text = unicodedata.normalize("NFKC", response)
    text = text.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    text = re.sub(r"</?(?:think|thinking|reasoning|analysis)\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^|]+?\|>", " ", text)
    text = re.sub(r"[*_`]+", "", text)
    return text.rstrip()


def _normalize_label_token(token: str) -> str:
    s = unicodedata.normalize("NFKC", token).strip().lower()
    s = s.strip(" \t\r\n'\"`*_[](){}<>.,;:!?")
    s = re.sub(r"^(?:option|answer)\s*", "", s)
    return s.strip(" \t\r\n'\"`*_[](){}<>.,;:!?")


def _valid_options(num_options: int) -> Set[str]:
    return {chr(ord("a") + index) for index in range(num_options)}


def _valid_options_for_style(num_options: int, label_style: Optional[str]) -> Set[str]:
    if label_style == "numbers":
        return {str(index + 1) for index in range(num_options)}
    return _valid_options(num_options)


def _extract_valid_matches(pattern: str, text: str, valid: Set[str]) -> list[str]:
    matches = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        token = _normalize_label_token(match.group(1))
        if token in valid:
            matches.append(token)
    return matches


def _last_valid_match(pattern: str, text: str, valid: Set[str]) -> Optional[str]:
    matches = _extract_valid_matches(pattern, text, valid)
    return matches[-1] if matches else None


def infer_option_label_style(prompt_text: str, num_options: int) -> Optional[str]:
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return None

    normalized = unicodedata.normalize("NFKC", prompt_text).lower()
    max_labels = min(num_options, 4)
    number_hits = 0
    letter_hits = 0

    for index in range(max_labels):
        number_label = str(index + 1)
        letter_label = chr(ord("a") + index)

        number_patterns = [
            rf"(?:^|\n)\s*\(\s*{re.escape(number_label)}\s*\)",
            rf"(?:^|\n)\s*{re.escape(number_label)}[\)\.\:]\s+",
        ]
        letter_patterns = [
            rf"(?:^|\n)\s*\(\s*{re.escape(letter_label)}\s*\)",
            rf"(?:^|\n)\s*{re.escape(letter_label)}[\)\.\:]\s+",
        ]

        if any(re.search(pattern, normalized, flags=re.MULTILINE) for pattern in number_patterns):
            number_hits += 1
        if any(re.search(pattern, normalized, flags=re.MULTILINE) for pattern in letter_patterns):
            letter_hits += 1

    if number_hits >= 2 and number_hits > letter_hits:
        return "numbers"
    if letter_hits >= 2 and letter_hits > number_hits:
        return "letters"
    return None


def extract_choice_with_strategy(
    response: str,
    num_options: int,
    label_style: Optional[str] = None,
) -> ChoiceParseResult:
    if not isinstance(response, str) or not response.strip():
        return ChoiceParseResult(choice=None, strategy=None)

    text = _normalize_response_text(response).lower()
    tail = text[-3000:] if len(text) > 3000 else text
    valid = _valid_options(num_options)

    explicit_marker = (
        r"(?:final\s+answer|final|answer|my\s+answer|choice|"
        r"chosen\s+(?:option|answer)|selected\s+(?:option|answer))"
    )

    short_line_patterns = [
        rf"(?:{explicit_marker})?\s*[:\-]?\s*(?:is\s+)?(?:option\s*)?[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?\.?",
        r"[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?\.?",
    ]

    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    for line in reversed(lines[-8:]):
        if len(line) > 120:
            continue
        for pattern in short_line_patterns:
            match = re.fullmatch(pattern, line, flags=re.IGNORECASE)
            if not match:
                continue
            token = _normalize_label_token(match.group(1))
            if token in valid:
                return ChoiceParseResult(choice=token, strategy="short_answer_line")

    answer_choice = _last_valid_match(
        rf"{explicit_marker}\s*[:\-]?\s*(?:is\s+)?(?:option\s*)?[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?",
        tail,
        valid,
    )
    if answer_choice:
        return ChoiceParseResult(choice=answer_choice, strategy="answer_marker")

    decision_choice = _last_valid_match(
        r"\bi(?:'d|'ll)?\s+(?:would\s+|will\s+|should\s+)?"
        r"(?:choose|select|pick|go\s+with|opt\s+for|prefer)\s+"
        r"(?:option\s*)?[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?",
        tail,
        valid,
    )
    if decision_choice:
        return ChoiceParseResult(choice=decision_choice, strategy="decision_verb")

    best_option_choice = _last_valid_match(
        r"(?:option\s*)?[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?\s+"
        r"(?:is|looks|seems|would\s+be|offers|has|gives|yields|provides)\s+"
        r"(?:the\s+)?(?:best|better|highest|highest-scoring|highest\s+scoring|"
        r"strongest|preferred|right|correct)\b",
        tail,
        valid,
    )
    if best_option_choice:
        return ChoiceParseResult(choice=best_option_choice, strategy="best_option_statement")

    best_choice_choice = _last_valid_match(
        r"(?:best|better|preferred|right|correct)\s+(?:choice|option|answer)\s+"
        r"(?:is\s+)?(?:option\s*)?[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?",
        tail,
        valid,
    )
    if best_choice_choice:
        return ChoiceParseResult(choice=best_choice_choice, strategy="best_choice_statement")

    trailing_token_match = re.search(
        r"(?:^|[\s\(\[\{])([a-z0-9]+)[\s\)\]\}\.,;:!?\"']*$",
        tail,
        flags=re.IGNORECASE,
    )
    if trailing_token_match:
        token = _normalize_label_token(trailing_token_match.group(1))
        if token in valid:
            return ChoiceParseResult(choice=token, strategy="trailing_token")

    compact = _normalize_label_token(re.sub(r"\s+", "", text))
    styled_valid = _valid_options_for_style(num_options, label_style)
    if compact in styled_valid:
        return ChoiceParseResult(choice=compact, strategy="bare_token")

    return ChoiceParseResult(choice=None, strategy=None)
