import difflib
import hashlib
import uuid
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def extract_corrected_samples(
    image_path: Path,
    detections: list[dict[str, Any]],
    original_text: str,
    edited_text: str,
    output_dir: Path,
    engine: str,
) -> list[dict[str, Any]]:
    lines = [str(detection.get("text", "")) for detection in detections]
    if not lines or "\n".join(lines) != original_text:
        return []

    matcher = difflib.SequenceMatcher(None, original_text, edited_text, autojunk=False)
    changed_groups: list[list[int]] = []
    for tag, start, end, _, _ in matcher.get_opcodes():
        if tag == "equal":
            continue
        affected_lines = sorted(_lines_for_change(start, end, lines))
        if affected_lines:
            changed_groups.append(affected_lines)

    if not changed_groups:
        return []

    groups = _merge_overlapping_groups(changed_groups)
    spans = _line_spans(lines)
    equal_blocks = matcher.get_matching_blocks()
    sample_dir = output_dir / "training_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        for group in groups:
            first, last = group[0], group[-1]
            output_start = _output_boundary(
                equal_blocks,
                matcher.get_opcodes(),
                spans[first][0],
                from_left=True,
                edited_length=len(edited_text),
            )
            output_end = _output_boundary(
                equal_blocks,
                matcher.get_opcodes(),
                spans[last][1],
                from_left=False,
                edited_length=len(edited_text),
            )
            corrected_label = edited_text[output_start:output_end].strip()
            if not corrected_label:
                continue

            boxes = [
                detections[index].get("box")
                for index in group
                if detections[index].get("box")
            ]
            if not boxes:
                continue
            points = [point for box in boxes for point in box]
            left = min(float(point[0]) for point in points)
            top = min(float(point[1]) for point in points)
            right = max(float(point[0]) for point in points)
            bottom = max(float(point[1]) for point in points)
            padding_x = max(6, int((right - left) * 0.06))
            padding_y = max(6, int((bottom - top) * 0.15))
            bounds = (
                max(0, int(left) - padding_x),
                max(0, int(top) - padding_y),
                min(image.width, int(right + 0.5) + padding_x),
                min(image.height, int(bottom + 0.5) + padding_y),
            )
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                continue

            sample_id = str(uuid.uuid4())
            relative_path = Path("training_samples") / f"{sample_id}.png"
            crop = image.crop(bounds)
            crop.save(sample_dir / f"{sample_id}.png", format="PNG")
            fingerprint = hashlib.sha256(
                f"{crop.width}x{crop.height}:RGB:".encode("ascii")
                + crop.tobytes()
            ).hexdigest()
            samples.append(
                {
                    "id": sample_id,
                    "image_path": str(relative_path),
                    "fingerprint": fingerprint,
                    "engine": engine,
                    "label": corrected_label,
                    "line_start": first,
                    "line_end": last,
                }
            )

    return samples


def _line_spans(lines: list[str]) -> list[tuple[int, int]]:
    spans = []
    offset = 0
    for index, line in enumerate(lines):
        start = offset
        end = start + len(line)
        if index < len(lines) - 1:
            end += 1
        spans.append((start, end))
        offset = end
    return spans


def _lines_for_change(start: int, end: int, lines: list[str]) -> set[int]:
    spans = _line_spans(lines)
    if start == end:
        for index, (line_start, line_end) in enumerate(spans):
            if line_start <= start < line_end or (
                index == len(spans) - 1 and start == line_end
            ):
                return {index}
        return {len(spans) - 1}
    changed = {
        index
        for index, (line_start, line_end) in enumerate(spans)
        if start < line_end and end > line_start
    }
    return changed


def _merge_overlapping_groups(candidates: list[list[int]]) -> list[list[int]]:
    groups: list[list[int]] = []
    for candidate in sorted(candidates, key=lambda group: group[0]):
        if not groups or candidate[0] > groups[-1][-1]:
            groups.append(candidate)
        else:
            groups[-1] = sorted(set(groups[-1]).union(candidate))
    return groups


def _output_boundary(
    matching_blocks: list[Any],
    opcodes: list[tuple[str, int, int, int, int]],
    original_boundary: int,
    *,
    from_left: bool,
    edited_length: int,
) -> int:
    insertions = [
        (target_start, target_end)
        for tag, source_start, source_end, target_start, target_end in opcodes
        if tag == "insert" and source_start == source_end == original_boundary
    ]
    if insertions:
        return min(start for start, _ in insertions) if from_left else max(
            end for _, end in insertions
        )

    containing = [
        block
        for block in matching_blocks
        if block.size and block.a <= original_boundary <= block.a + block.size
    ]
    if containing:
        block = (
            max(containing, key=lambda item: item.a)
            if from_left
            else min(containing, key=lambda item: item.a)
        )
        return block.b + original_boundary - block.a

    if from_left:
        anchors = [
            block
            for block in matching_blocks
            if block.size and block.a + block.size <= original_boundary
        ]
        if not anchors:
            return 0
        anchor = max(anchors, key=lambda block: block.a + block.size)
        return anchor.b + anchor.size

    anchors = [
        block
        for block in matching_blocks
        if block.size and block.a >= original_boundary
    ]
    if not anchors:
        return edited_length
    anchor = min(anchors, key=lambda block: block.a)
    return anchor.b
