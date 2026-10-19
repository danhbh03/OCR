import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any


def merge_pdf_text(
    page: Any,
    detections: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    pdf_lines = _extract_pdf_lines(page)
    if not pdf_lines or not detections:
        return _join_detection_text(detections), detections

    candidate_matches = []
    for detection_index, detection in enumerate(detections):
        detection_text = str(detection.get("text", "")).strip()
        if not detection_text:
            continue
        normalized_detection = _normalize_for_comparison(detection_text)
        if not normalized_detection:
            continue
        for pdf_index, pdf_line in enumerate(pdf_lines):
            similarity = SequenceMatcher(
                None,
                normalized_detection,
                pdf_line["normalized_text"],
                autojunk=False,
            ).ratio()
            if similarity >= 0.55:
                candidate_matches.append((similarity, detection_index, pdf_index))

    candidate_matches.sort(reverse=True)
    matched_detections: set[int] = set()
    matched_pdf_lines: set[int] = set()
    merged_detections = [dict(detection) for detection in detections]
    for _, detection_index, pdf_index in candidate_matches:
        if detection_index in matched_detections or pdf_index in matched_pdf_lines:
            continue
        matched_detections.add(detection_index)
        matched_pdf_lines.add(pdf_index)
        merged_detections[detection_index]["text"] = pdf_lines[pdf_index]["text"]
        merged_detections[detection_index]["text_source"] = "pdf"

    return _join_detection_text(merged_detections), merged_detections


def _extract_pdf_lines(page: Any) -> list[dict[str, str]]:
    grouped_words: dict[tuple[int, int], list[tuple[int, str]]] = {}

    for word in page.get_text("words", sort=False):
        if len(word) < 8:
            continue
        text, block_number, line_number, word_number = word[4:8]
        normalized = _normalize_for_comparison(str(text))
        if not normalized:
            continue
        key = (int(block_number), int(line_number))
        grouped_words.setdefault(key, []).append((int(word_number), str(text)))

    lines = []
    for words in grouped_words.values():
        words.sort(key=lambda word: word[0])
        text = _join_pdf_words([word[1] for word in words])
        lines.append(
            {
                "text": text,
                "normalized_text": _normalize_for_comparison(text),
            }
        )
    return lines


def _join_pdf_words(words: list[str]) -> str:
    text = " ".join(words)
    return re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)


def _normalize_for_comparison(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character) and character.isalnum()
    )


def _join_detection_text(detections: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(detection.get("text", "")).strip()
        for detection in detections
        if str(detection.get("text", "")).strip()
    )
