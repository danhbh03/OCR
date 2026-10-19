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
            matched_text, similarity = _best_matching_segment(
                normalized_detection, pdf_line["words"]
            )
            if similarity >= 0.55:
                candidate_matches.append(
                    (similarity, detection_index, pdf_index, matched_text)
                )

    candidate_matches.sort(reverse=True)
    matched_detections: set[int] = set()
    matched_pdf_lines: set[int] = set()
    merged_detections = [dict(detection) for detection in detections]
    for _, detection_index, pdf_index, matched_text in candidate_matches:
        if detection_index in matched_detections or pdf_index in matched_pdf_lines:
            continue
        matched_detections.add(detection_index)
        matched_pdf_lines.add(pdf_index)
        merged_detections[detection_index]["text"] = matched_text
        merged_detections[detection_index]["text_source"] = "pdf"

    return _join_detection_text(merged_detections), merged_detections


def _extract_pdf_lines(page: Any) -> list[dict[str, Any]]:
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
        lines.append(
            {
                "words": [word[1] for word in words],
            }
        )
    return lines


def _best_matching_segment(
    normalized_detection: str,
    pdf_words: list[str],
) -> tuple[str, float]:
    best_text = ""
    best_similarity = 0.0
    best_length_difference = float("inf")
    for start in range(len(pdf_words)):
        for end in range(start + 1, len(pdf_words) + 1):
            candidate_text = _join_pdf_words(pdf_words[start:end])
            normalized_candidate = _normalize_for_comparison(candidate_text)
            similarity = SequenceMatcher(
                None,
                normalized_detection,
                normalized_candidate,
                autojunk=False,
            ).ratio()
            length_difference = abs(
                len(normalized_detection) - len(normalized_candidate)
            )
            if (
                similarity > best_similarity
                or (
                    similarity == best_similarity
                    and length_difference < best_length_difference
                )
            ):
                best_text = candidate_text
                best_similarity = similarity
                best_length_difference = length_difference
    return best_text, best_similarity


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
