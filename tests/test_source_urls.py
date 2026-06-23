from __future__ import annotations

from src.utils.helpers import parse_source_urls


def test_parse_source_urls_splits_semicolon_delimited_values():
    urls = parse_source_urls("https://one.example; https://two.example")

    assert urls == ["https://one.example", "https://two.example"]


def test_parse_source_urls_deduplicates_values():
    urls = parse_source_urls("https://one.example; https://one.example\nhttps://two.example")

    assert urls == ["https://one.example", "https://two.example"]


def test_parse_source_urls_reads_json_lists():
    urls = parse_source_urls('["https://one.example", "https://two.example"]')

    assert urls == ["https://one.example", "https://two.example"]


def test_parse_source_urls_extracts_encoded_semicolon_joined_urls():
    urls = parse_source_urls(
        "https://jcapasia.com;%20https//hk.linkedin.com/company/jcapasia;%20"
        "https://en.wikipedia.org/wiki/Jebsen_Group"
    )

    assert urls == [
        "https://jcapasia.com",
        "https://hk.linkedin.com/company/jcapasia",
        "https://en.wikipedia.org/wiki/Jebsen_Group",
    ]


def test_parse_source_urls_ignores_non_urls():
    assert parse_source_urls("not a source; also not a source") == []
