from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import unquote


URL_PATTERN = re.compile(r"https?:?/?/?[^\s,;]+")


def compact_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def normalize_source_url(value: Any) -> str:
    text = unquote(compact_str(value)).strip().strip("[](){}\"'.,")
    if not text:
        return ""
    text = re.sub(r"^https:/([^/])", r"https://\1", text)
    text = re.sub(r"^http:/([^/])", r"http://\1", text)
    text = re.sub(r"^https//", "https://", text)
    text = re.sub(r"^http//", "http://", text)
    if text.startswith("www."):
        text = f"https://{text}"
    if not re.match(r"^https?://", text):
        return ""
    return text


def parse_source_urls(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [compact_str(item) for item in value if compact_str(item)]
    text = compact_str(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [compact_str(item) for item in parsed if compact_str(item)]
    if isinstance(parsed, str):
        text = parsed

    urls: list[str] = []
    candidates = URL_PATTERN.findall(text)
    if not candidates:
        candidates = re.split(r"[\n,;]+", text)
    for part in candidates:
        url = normalize_source_url(part)
        if url and url not in urls:
            urls.append(url)
    return urls


def first_non_empty(values: Iterable[Any]) -> str:
    for value in values:
        text = compact_str(value)
        if text:
            return text
    return ""
