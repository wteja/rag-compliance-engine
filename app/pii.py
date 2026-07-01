from __future__ import annotations

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()


def redact_with_counts(text: str) -> tuple[str, dict[str, int]]:
    results = _analyzer.analyze(text=text, language="en")
    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
    counts: dict[str, int] = {}
    for item in anonymized.items:
        counts[item.entity_type] = counts.get(item.entity_type, 0) + 1
    return anonymized.text, counts


def redact(text: str) -> str:
    return redact_with_counts(text)[0]
