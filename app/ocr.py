import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


class OCRUnavailableError(RuntimeError):
    pass


logger = logging.getLogger(__name__)

DEWARP_STRIP_COUNT = 32
DEWARP_MIN_TEXT_LINES = 4
DEWARP_MAX_DISPLACEMENT_RATIO = 0.025
DEWARP_MIN_DISPLACEMENT_RATIO = 0.002
DEWARP_SMOOTHING_RATIO = 0.08
DEWARP_THRESHOLD_BLOCK_SIZE = 31
DEWARP_THRESHOLD_C = 15
DEWARP_MIN_PAGE_DIMENSION = 160
DEWARP_PROJECTION_KERNEL_SIZE = 3
DEWARP_MIN_INK_WIDTH_RATIO = 0.003
DEWARP_PEAK_PROJECTION_RATIO = 0.08
DEWARP_MIN_LINE_BAND_HEIGHT = 3
DEWARP_MAX_LINE_BAND_HEIGHT_RATIO = 0.1
DEWARP_STRIP_INK_RATIO = 0.015
DEWARP_MIN_VALID_STRIP_RATIO = 0.5
DEWARP_MIN_VALID_STRIPS = 8
DEWARP_MIN_THRESHOLD_BLOCK_SIZE = 3
DEWARP_MIN_MAX_LINE_BAND_HEIGHT = 8
DEWARP_MIN_SMOOTHING_WIDTH = 3
ORIENTATION_MIN_CONFIDENCE = 5.0
DESKEW_MIN_ANGLE = 0.4
DESKEW_MAX_ANGLE = 80.0


@dataclass(frozen=True)
class PageTransform:
    matrix: np.ndarray
    dewarp_offsets: np.ndarray | None = None

def recognize(image_path: str, engine: str) -> tuple[str, list[dict[str, Any]]]:
    if engine not in {
        "ppocrv6",
        "paddle",
        "tesseract",
        "doclayout_tesseract",
        "doclayout_ppocr_tesseract",
    }:
        raise ValueError(f"OCR engine không hợp lệ: {engine}")

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    aligned_image, image_to_aligned = _align_page(image)

    if engine in {"ppocrv6", "paddle", "doclayout_ppocr_tesseract"}:
        tesseract = _get_tesseract()
        layout_regions = []
        layout_blocks = []
        try:
            if engine == "doclayout_ppocr_tesseract":
                layout_blocks = _detect_layout_blocks(aligned_image)
                layout_regions = [
                    (x, y, width, height)
                    for x, y, width, height, _ in layout_blocks
                ]
            else:
                _get_layout_detector()
                layout_regions = _detect_layout_regions(aligned_image)
        except (OCRUnavailableError, OSError, RuntimeError, ValueError) as error:
            logger.warning(
                "Không chạy được DocLayout-YOLO; sẽ sắp dòng PP-OCR theo "
                "tọa độ: %s",
                error,
            )
        if engine == "doclayout_ppocr_tesseract":
            logger.info(
                "Chế độ riêng: DocLayout sắp xếp khối, PP-OCR phát hiện dòng, "
                "Tesseract nhận dạng."
            )
        if engine == "doclayout_ppocr_tesseract" and any(
            label == "table" for _, _, _, _, label in layout_blocks
        ):
            lines = _recognize_tesseract_layout_with_tables(
                tesseract,
                aligned_image,
                _order_layout_blocks(layout_blocks),
                use_ppocr_for_text_blocks=True,
            )
            _map_lines_to_original(lines, image_to_aligned, image.size)
            return _join_lines(lines), lines
        if layout_regions:
            lines = _recognize_tesseract_layout_ppocr_regions(
                tesseract,
                aligned_image,
                _order_layout_regions(layout_regions),
            )
            _map_lines_to_original(lines, image_to_aligned, image.size)
            return _join_lines(lines), lines
        text_regions = _detect_text_regions(aligned_image)
        if text_regions:
            lines = _recognize_tesseract_text_regions(
                tesseract,
                aligned_image,
                _order_text_regions(text_regions, layout_regions),
            )
            if not lines:
                lines = _recognize_tesseract_page(tesseract, aligned_image)
        else:
            lines = _recognize_tesseract_page(tesseract, aligned_image)
        _map_lines_to_original(lines, image_to_aligned, image.size)
        return _join_lines(lines), lines

    tesseract = _get_tesseract()
    if engine == "doclayout_tesseract":
        try:
            layout_blocks = _detect_layout_blocks(aligned_image)
        except (OCRUnavailableError, OSError, RuntimeError, ValueError) as error:
            logger.warning(
                "Không chạy được DocLayout-YOLO; chế độ tự chọn sẽ dùng "
                "PP-OCR trên toàn trang: %s",
                error,
            )
            layout_blocks = []

        ordered_blocks = _order_layout_blocks(layout_blocks)
        table_blocks = [
            block for block in ordered_blocks if block[4] == "table"
        ]
        if table_blocks:
            text_blocks = [
                block for block in ordered_blocks if block[4] != "table"
            ]
            lines = _recognize_tesseract_layout_with_tables(
                tesseract,
                aligned_image,
                ordered_blocks,
                use_ppocr_for_text_blocks=(
                    bool(text_blocks)
                    and _should_use_ppocr_for_layout_blocks(
                        text_blocks, aligned_image.size
                    )
                ),
            )
            _map_lines_to_original(lines, image_to_aligned, image.size)
            return _join_lines(lines), lines

        layout_regions = [
            (x, y, width, height)
            for x, y, width, height, _ in ordered_blocks
        ]
        if layout_blocks and _should_use_ppocr_for_layout_blocks(
            layout_blocks, aligned_image.size
        ):
            logger.info(
                "Bố cục nhiều cột/nhiều khối: dùng DocLayout sắp xếp, "
                "PP-OCR phát hiện dòng và Tesseract nhận dạng."
            )
            if layout_regions:
                lines = _recognize_tesseract_layout_ppocr_regions(
                    tesseract,
                    aligned_image,
                    layout_regions,
                )
            else:
                text_regions = _detect_text_regions(aligned_image)
                if text_regions:
                    lines = _recognize_tesseract_text_regions(
                        tesseract,
                        aligned_image,
                        _order_text_regions(text_regions, []),
                    )
                else:
                    lines = _recognize_tesseract_page(tesseract, aligned_image)
        elif layout_blocks:
            logger.info(
                "Bố cục chủ yếu là các khối văn bản lớn: dùng DocLayout chia "
                "khối rồi Tesseract đọc trực tiếp."
            )
            lines = _recognize_tesseract_layout_blocks(
                tesseract,
                aligned_image,
                ordered_blocks,
            )
        else:
            logger.info(
                "Không có khối DocLayout; dùng PP-OCR phát hiện dòng trên "
                "toàn trang."
            )
            text_regions = _detect_text_regions(aligned_image)
            if text_regions:
                lines = _recognize_tesseract_text_regions(
                    tesseract,
                    aligned_image,
                    _order_text_regions(text_regions, []),
                )
                if not lines:
                    lines = _recognize_tesseract_page(tesseract, aligned_image)
            else:
                lines = _recognize_tesseract_page(tesseract, aligned_image)
        _map_lines_to_original(lines, image_to_aligned, image.size)
        return _join_lines(lines), lines

    regions = _detect_layout_regions(aligned_image)
    if not regions:
        lines = _recognize_tesseract_page(tesseract, aligned_image)
    else:
        lines = []
        for x, y, width, height in regions:
            text_height = max(8.0, min(width, height) * 0.06)
            crop_box = (
                max(0, x - round(text_height)),
                max(0, y - round(text_height)),
                min(aligned_image.width, x + width + round(text_height)),
                min(aligned_image.height, y + height + round(text_height)),
            )
            crop = aligned_image.crop(crop_box)
            crop, scale = _preprocess_tesseract_region(crop, text_height)
            if crop is None:
                continue
            page_segmentation_mode = 11 if y < aligned_image.height * 0.12 else 6
            region_lines = _recognize_tesseract_page(
                tesseract, crop, page_segmentation_mode
            )
            for line in region_lines:
                for point in line["box"]:
                    point[0] = point[0] / scale + crop_box[0]
                    point[1] = point[1] / scale + crop_box[1]
                lines.append(line)
    _map_lines_to_original(lines, image_to_aligned, image.size)
    return _join_lines(lines), lines


def _detect_layout_regions(
    image: Image.Image,
) -> list[tuple[int, int, int, int]]:
    return [
        (x, y, width, height)
        for x, y, width, height, _ in _detect_layout_blocks(image)
    ]


def _detect_layout_blocks(
    image: Image.Image,
) -> list[tuple[int, int, int, int, str]]:
    model, device = _get_layout_detector()
    image_array = cv2.cvtColor(
        np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR
    )
    results = model.predict(
        image_array,
        imgsz=1024,
        conf=0.2,
        device=device,
        verbose=False,
    )
    if not results:
        return []

    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    coordinates = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()
    names = result.names
    detections = [
        (
            *box,
            confidence,
            names[int(class_id)],
        )
        for box, confidence, class_id in zip(
            coordinates, confidences, class_ids
        )
    ]
    return _layout_blocks_from_detections(detections, image.size)


def _detect_text_regions(image: Image.Image) -> list[dict[str, Any]]:
    detector = _get_text_detector()
    image_array = cv2.cvtColor(
        np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR
    )
    try:
        results = detector.predict(image_array, batch_size=1)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise OCRUnavailableError(
            f"PP-OCRv6 Medium không thể phát hiện vùng chữ: {error}"
        ) from error

    regions = []
    for result in results:
        result_json = result.json
        detection_result = result_json.get("res", result_json)
        polygons = detection_result.get("dt_polys", [])
        scores = detection_result.get("dt_scores", [])
        for index, polygon in enumerate(polygons):
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) < 4 or not np.isfinite(points).all():
                continue
            points[:, 0] = np.clip(points[:, 0], 0, image.width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, image.height - 1)
            score = float(scores[index]) if index < len(scores) else None
            regions.append({"box": points.tolist(), "confidence": score})
    return regions


def _layout_regions_from_detections(
    detections: list[tuple[float, float, float, float, float, str]],
    image_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    return [
        (x, y, width, height)
        for x, y, width, height, _ in _layout_blocks_from_detections(
            detections, image_size
        )
    ]


def _layout_blocks_from_detections(
    detections: list[tuple[float, float, float, float, float, str]],
    image_size: tuple[int, int],
) -> list[tuple[int, int, int, int, str]]:
    image_width, image_height = image_size
    text_labels = {
        "title",
        "text",
        "list item",
        "section header",
        "page header",
        "page footer",
        "table",
        "caption",
        "footnote",
        "formula",
    }
    regions = []
    for left, top, right, bottom, confidence, label in detections:
        normalized_label = (
            label.strip().lower().replace("_", " ").replace("-", " ")
        )
        if confidence < 0.2 or normalized_label not in text_labels:
            continue
        x1 = max(0, min(image_width, round(left)))
        y1 = max(0, min(image_height, round(top)))
        x2 = max(0, min(image_width, round(right)))
        y2 = max(0, min(image_height, round(bottom)))
        if x2 <= x1 or y2 <= y1:
            continue
        regions.append((x1, y1, x2 - x1, y2 - y1, normalized_label))

    regions = list(dict.fromkeys(regions))
    deduplicated = []
    for region in sorted(regions, key=lambda item: item[2] * item[3], reverse=True):
        x, y, width, height, _ = region
        area = width * height
        is_duplicate = False
        for other in deduplicated:
            overlap_width = max(
                0,
                min(x + width, other[0] + other[2]) - max(x, other[0]),
            )
            overlap_height = max(
                0,
                min(y + height, other[1] + other[3]) - max(y, other[1]),
            )
            intersection = overlap_width * overlap_height
            union = area + other[2] * other[3] - intersection
            if intersection / max(1, union) >= 0.88:
                is_duplicate = True
                break
        if is_duplicate:
            continue
        deduplicated.append(region)
    regions = deduplicated
    regions = [
        region
        for region in regions
        if not any(
            other is not region
            and other[2] * other[3] > region[2] * region[3]
            and other[0] <= region[0]
            and other[1] <= region[1]
            and other[0] + other[2] >= region[0] + region[2]
            and other[1] + other[3] >= region[1] + region[3]
            for other in regions
        )
    ]
    regions = _expand_layout_blocks_vertically(regions, image_height)
    return sorted(regions, key=lambda region: (region[1], region[0]))


def _expand_layout_blocks_vertically(
    blocks: list[tuple[int, int, int, int, str]],
    image_height: int,
) -> list[tuple[int, int, int, int, str]]:
    for padding in (3, 2, 1):
        expanded = [
            (
                x,
                max(0, y - padding),
                width,
                min(image_height, y + height + padding) - max(0, y - padding),
                label,
            )
            for x, y, width, height, label in blocks
        ]
        creates_overlap = False
        for index, (x, y, width, height, _) in enumerate(blocks):
            for other_index, (
                other_x,
                other_y,
                other_width,
                other_height,
                _,
            ) in enumerate(blocks[index + 1 :], start=index + 1):
                if x >= other_x + other_width or other_x >= x + width:
                    continue
                if y < other_y + other_height and other_y < y + height:
                    continue
                expanded_y = expanded[index][1]
                expanded_bottom = expanded_y + expanded[index][3]
                other_expanded_y = expanded[other_index][1]
                other_expanded_bottom = (
                    other_expanded_y + expanded[other_index][3]
                )
                if expanded_y < other_expanded_bottom and other_expanded_y < expanded_bottom:
                    creates_overlap = True
                    break
            if creates_overlap:
                break
        if not creates_overlap:
            return expanded
    return blocks


def _order_layout_regions(
    regions: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    text_regions = [
        {
            "text": "",
            "box": [
                [x, y],
                [x + width, y],
                [x + width, y + height],
                [x, y + height],
            ],
            "layout_index": index,
        }
        for index, (x, y, width, height) in enumerate(regions)
    ]
    ordered = _order_text_regions(text_regions, [])
    return [regions[region["layout_index"]] for region in ordered]


def _order_layout_blocks(
    blocks: list[tuple[int, int, int, int, str]],
) -> list[tuple[int, int, int, int, str]]:
    ordered_regions = _order_layout_regions(
        [(x, y, width, height) for x, y, width, height, _ in blocks]
    )
    remaining = list(blocks)
    ordered = []
    for region in ordered_regions:
        index = next(
            index
            for index, block in enumerate(remaining)
            if block[:4] == region
        )
        ordered.append(remaining.pop(index))
    return ordered


def _merge_layout_headings(
    blocks: list[tuple[int, int, int, int, str]],
) -> list[tuple[int, int, int, int, str]]:
    merged = list(blocks)
    consumed: set[int] = set()
    for heading_index, (
        heading_x,
        heading_y,
        heading_width,
        heading_height,
        label,
    ) in enumerate(merged):
        if label not in {"title", "section header"} or heading_index in consumed:
            continue

        heading_bottom = heading_y + heading_height
        max_gap = max(36, round(heading_height * 1.5))
        candidates = []
        for text_index, (
            text_x,
            text_y,
            text_width,
            text_height,
            text_label,
        ) in enumerate(merged):
            if (
                text_index == heading_index
                or text_index in consumed
                or text_label != "text"
            ):
                continue
            vertical_gap = text_y - heading_bottom
            left_alignment = abs(text_x - heading_x) <= max(
                12, round(heading_height * 0.5)
            )
            horizontal_overlap = max(
                0,
                min(heading_x + heading_width, text_x + text_width)
                - max(heading_x, text_x),
            )
            overlap_ratio = horizontal_overlap / max(
                1, min(heading_width, text_width)
            )
            heading_is_within_text_width = (
                text_x <= heading_x
                and text_x + text_width >= heading_x + heading_width
            )
            if (
                0 <= vertical_gap <= max_gap
                and (left_alignment or heading_is_within_text_width)
                and overlap_ratio >= 0.8
            ):
                candidates.append(
                    (vertical_gap, text_index, text_width, text_height)
                )
        if not candidates:
            continue

        _, text_index, text_width, text_height = min(candidates)
        text_x, text_y = merged[text_index][0:2]
        left = min(heading_x, text_x)
        top = min(heading_y, text_y)
        right = max(heading_x + heading_width, text_x + text_width)
        bottom = max(heading_bottom, text_y + text_height)
        merged[text_index] = (
            left,
            top,
            right - left,
            bottom - top,
            "text",
        )
        consumed.add(heading_index)

    return [block for index, block in enumerate(merged) if index not in consumed]


def _should_use_ppocr_for_layout_blocks(
    blocks: list[tuple[int, int, int, int, str]],
    image_size: tuple[int, int],
) -> bool:
    image_width, _ = image_size
    minimum_column_gap = max(12, round(image_width * 0.08))
    for index, (x1, y1, width1, height1, _) in enumerate(blocks):
        for x2, y2, width2, height2, _ in blocks[index + 1 :]:
            vertical_overlap = max(
                0,
                min(y1 + height1, y2 + height2) - max(y1, y2),
            )
            overlap_ratio = vertical_overlap / max(1, min(height1, height2))
            horizontal_gap = max(
                x2 - (x1 + width1),
                x1 - (x2 + width2),
                0,
            )
            if overlap_ratio >= 0.5 and horizontal_gap >= minimum_column_gap:
                return True
    return False


def _recognize_tesseract_layout_blocks(
    tesseract: Any,
    image: Image.Image,
    blocks: list[tuple[int, int, int, int, str]],
) -> list[dict[str, Any]]:
    lines = []
    for x, y, width, height, _ in _merge_layout_headings(blocks):
        text_height = max(8.0, min(width, height) * 0.06)
        crop_box = (
            max(0, x - round(text_height)),
            max(0, y - round(text_height)),
            min(image.width, x + width + round(text_height)),
            min(image.height, y + height + round(text_height)),
        )
        crop = image.crop(crop_box)
        crop, dewarp_offsets = _dewarp_page(crop)
        dewarped_size = crop.size
        crop, scale = _preprocess_tesseract_region(crop, text_height)
        if crop is None:
            continue
        page_segmentation_mode = 11 if y < image.height * 0.12 else 6
        block_lines = _recognize_tesseract_page(
            tesseract, crop, page_segmentation_mode
        )
        for line in block_lines:
            for point in line["box"]:
                point[0] /= scale
                point[1] /= scale
        if dewarp_offsets is not None:
            _map_lines_to_original(
                block_lines,
                PageTransform(np.eye(3, dtype=np.float32), dewarp_offsets),
                dewarped_size,
            )
        for line in block_lines:
            for point in line["box"]:
                point[0] += crop_box[0]
                point[1] += crop_box[1]
            lines.append(line)
    return lines


def _recognize_tesseract_layout_with_tables(
    tesseract: Any,
    image: Image.Image,
    blocks: list[tuple[int, int, int, int, str]],
    use_ppocr_for_text_blocks: bool,
) -> list[dict[str, Any]]:
    lines = []
    text_blocks = []

    def flush_text_blocks() -> None:
        if not text_blocks:
            return
        if use_ppocr_for_text_blocks:
            regions = [
                (x, y, width, height)
                for x, y, width, height, _ in text_blocks
            ]
            lines.extend(
                _recognize_tesseract_layout_ppocr_regions(
                    tesseract, image, regions
                )
            )
        else:
            lines.extend(
                _recognize_tesseract_layout_blocks(
                    tesseract, image, text_blocks
                )
            )
        text_blocks.clear()

    for block in blocks:
        if block[4] != "table":
            text_blocks.append(block)
            continue
        flush_text_blocks()
        lines.extend(_recognize_tesseract_table_block(tesseract, image, block))

    flush_text_blocks()
    return lines


def _recognize_tesseract_table_block(
    tesseract: Any,
    image: Image.Image,
    block: tuple[int, int, int, int, str],
) -> list[dict[str, Any]]:
    x, y, width, height, _ = block
    padding = max(3, round(min(width, height) * 0.02))
    crop_box = (
        max(0, x - padding),
        max(0, y - padding),
        min(image.width, x + width + padding),
        min(image.height, y + height + padding),
    )
    crop = image.crop(crop_box)
    crop, dewarp_offsets = _dewarp_page(crop)
    dewarped_size = crop.size
    grid = _find_table_grid(crop)
    if grid is not None:
        try:
            text_regions = _detect_text_regions(crop)
        except (OCRUnavailableError, OSError, RuntimeError, ValueError) as error:
            logger.warning(
                "PP-OCR không phát hiện được dòng trong bảng; dùng Tesseract "
                "theo từng ô: %s",
                error,
            )
            text_regions = []
        if text_regions and _is_complex_table_grid(grid):
            logger.info(
                "Bảng có nhiều ô/đường gộp; nhận dạng theo dòng chữ để tránh "
                "áp dụng nhầm lưới cột cho các ô gộp."
            )
            ordered_regions = _order_table_text_regions_by_grid(
                text_regions, grid
            )
            table_lines = _recognize_tesseract_text_regions(
                tesseract,
                crop,
                ordered_regions,
                order_contact_values=False,
            )
            table_lines = _order_table_text_regions_by_grid(table_lines, grid)
        else:
            table_lines = _recognize_tesseract_table_grid(
                tesseract, crop, grid, text_regions
            )
    else:
        text_regions = _detect_text_regions(crop)
        if text_regions:
            ordered_regions = _order_table_text_regions(text_regions)
            table_lines = _recognize_tesseract_text_regions(
                tesseract,
                crop,
                ordered_regions,
                order_contact_values=False,
            )
            table_lines = _order_table_text_regions(table_lines)
        else:
            logger.warning(
                "Không tìm thấy lưới hoặc dòng chữ trong khung bảng "
                "(%d, %d, %d, %d); dùng Tesseract đọc toàn vùng.",
                x,
                y,
                width,
                height,
            )
            text_height = max(8.0, min(width, height) * 0.06)
            processed_crop, scale = _preprocess_tesseract_region(crop, text_height)
            if processed_crop is None:
                return []
            table_lines = _recognize_tesseract_page(
                tesseract, processed_crop, page_segmentation_mode=6
            )
            for line in table_lines:
                for point in line["box"]:
                    point[0] /= scale
                    point[1] /= scale

    if dewarp_offsets is not None:
        _map_lines_to_original(
            table_lines,
            PageTransform(np.eye(3, dtype=np.float32), dewarp_offsets),
            dewarped_size,
        )
    for line in table_lines:
        for point in line["box"]:
            point[0] += crop_box[0]
            point[1] += crop_box[1]
    return table_lines


def _is_complex_table_grid(
    grid: tuple[list[int], list[int]],
) -> bool:
    column_count = len(grid[0]) - 1
    row_count = len(grid[1]) - 1
    return column_count * row_count >= 72


def _order_table_text_regions_by_grid(
    text_regions: list[dict[str, Any]],
    grid: tuple[list[int], list[int]],
) -> list[dict[str, Any]]:
    column_lines, row_lines = grid

    def order_key(region: dict[str, Any]) -> tuple[int, int, float, float]:
        points = region.get("box", [])
        if not points:
            return len(row_lines), len(column_lines), 0.0, 0.0
        left = min(float(point[0]) for point in points)
        top = min(float(point[1]) for point in points)
        right = max(float(point[0]) for point in points)
        bottom = max(float(point[1]) for point in points)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        row_index = max(
            0,
            min(
                len(row_lines) - 2,
                int(np.searchsorted(row_lines, center_y, side="right") - 1),
            ),
        )
        column_index = max(
            0,
            min(
                len(column_lines) - 2,
                int(np.searchsorted(column_lines, center_x, side="right") - 1),
            ),
        )
        return row_index, column_index, top, left

    return sorted(text_regions, key=order_key)


def _find_table_grid(
    image: Image.Image,
) -> tuple[list[int], list[int]] | None:
    gray = cv2.cvtColor(
        np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY
    )
    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    height, width = binary.shape
    horizontal_kernel_size = max(12, round(width / 12))
    vertical_kernel_size = max(12, round(height / 12))
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (horizontal_kernel_size, 1)
        ),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, vertical_kernel_size)
        ),
    )

    def line_centers(
        mask: np.ndarray, axis: int, minimum_extent: float
    ) -> list[int]:
        coverage = (mask > 0).sum(axis=axis)
        positions = np.flatnonzero(coverage >= minimum_extent)
        groups: list[list[int]] = []
        for position in positions:
            value = int(position)
            if not groups or value > groups[-1][-1] + 1:
                groups.append([value])
            else:
                groups[-1].append(value)
        return [round(sum(group) / len(group)) for group in groups]

    row_lines = line_centers(horizontal, 1, width * 0.45)
    column_lines = line_centers(vertical, 0, height * 0.35)
    row_lines = [line for line in row_lines if 2 < line < height - 3]
    column_lines = [
        line for line in column_lines if 2 < line < width - 3
    ]
    if len(row_lines) < 1 or len(column_lines) < 1:
        return None
    return [0, *column_lines, width], [0, *row_lines, height]


def _recognize_tesseract_table_grid(
    tesseract: Any,
    image: Image.Image,
    grid: tuple[list[int], list[int]],
    text_regions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    column_lines, row_lines = grid
    cell_rows = []
    region_records = []
    for top, bottom in zip(row_lines, row_lines[1:]):
        row_cells = []
        for left, right in zip(column_lines, column_lines[1:]):
            inset = 3
            cell_box = (
                min(right, left + inset),
                min(bottom, top + inset),
                max(left, right - inset),
                max(top, bottom - inset),
            )
            cell_regions = []
            for region in text_regions or []:
                points = region.get("box", [])
                if not points:
                    continue
                center_x = sum(float(point[0]) for point in points) / len(points)
                center_y = sum(float(point[1]) for point in points) / len(points)
                if left <= center_x < right and top <= center_y < bottom:
                    cell_regions.append(region)
            if cell_regions:
                cell_image = image.crop(cell_box)
                cell_records = []
                for region in _order_table_text_regions(cell_regions):
                    local_region = {
                        **region,
                        "box": [
                            [
                                float(point[0]) - cell_box[0],
                                float(point[1]) - cell_box[1],
                            ]
                            for point in region["box"]
                        ],
                    }
                    recognized_lines = _recognize_tesseract_text_regions(
                        tesseract,
                        cell_image,
                        [local_region],
                        order_contact_values=False,
                    )
                    for line in recognized_lines:
                        for point in line["box"]:
                            point[0] += cell_box[0]
                            point[1] += cell_box[1]
                    record = {
                        "region": region,
                        "local_region": local_region,
                        "cell_box": cell_box,
                        "lines": recognized_lines,
                    }
                    cell_records.append(record)
                    region_records.append(record)
                cell_lines = [
                    line
                    for record in cell_records
                    for line in record["lines"]
                ]
            else:
                cell = image.crop(cell_box)
                cell_lines = _recognize_tesseract_page(
                    tesseract, cell, page_segmentation_mode=6
                )
                for line in cell_lines:
                    for point in line["box"]:
                        point[0] += cell_box[0]
                        point[1] += cell_box[1]
            row_cells.append(
                {"box": cell_box, "lines": cell_lines}
            )
        cell_rows.append(row_cells)

    numeric_region_ids = _numeric_table_region_ids(region_records)
    for record in region_records:
        if id(record["region"]) not in numeric_region_ids:
            continue
        left, top, _, _ = record["cell_box"]
        numeric_lines = _recognize_tesseract_numeric_region(
            tesseract,
            image.crop(record["cell_box"]),
            record["local_region"],
        )
        if not numeric_lines:
            continue
        for line in numeric_lines:
            for point in line["box"]:
                point[0] += left
                point[1] += top
        record["lines"] = numeric_lines

    lines = []
    for row in cell_rows:
        for cell in row:
            cell_regions = [
                record
                for record in region_records
                if record["cell_box"] == cell["box"]
            ]
            if cell_regions:
                cell["lines"] = [
                    line
                    for record in cell_regions
                    for line in record["lines"]
                ]
            lines.extend(cell["lines"])

    return lines


def _numeric_table_region_ids(
    region_records: list[dict[str, Any]],
) -> set[int]:
    candidates = []
    for record in region_records:
        points = record["region"].get("box", [])
        if not points:
            continue
        left = min(float(point[0]) for point in points)
        top = min(float(point[1]) for point in points)
        right = max(float(point[0]) for point in points)
        bottom = max(float(point[1]) for point in points)
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0 or width > height * 4:
            continue
        text = " ".join(
            str(line.get("text", "")).strip() for line in record["lines"]
        ).strip()
        candidates.append(
            {
            "id": id(record["region"]),
            "center": (left + right) / 2,
            "height": height,
            "text": text,
            }
        )
    if len(candidates) < 3:
        return set()

    heights = sorted(item["height"] for item in candidates)
    center_tolerance = max(8.0, heights[len(heights) // 2] * 0.9)
    columns: list[list[dict[str, Any]]] = []
    for candidate in sorted(candidates, key=lambda item: item["center"]):
        column_center = (
            sum(item["center"] for item in columns[-1]) / len(columns[-1])
            if columns
            else None
        )
        if (
            not columns
            or abs(candidate["center"] - column_center) > center_tolerance
        ):
            columns.append([candidate])
        else:
            columns[-1].append(candidate)

    numeric_region_ids = set()
    for column in columns:
        values = [item for item in column if item["text"]]
        numeric_values = [
            item
            for item in values
            if re.fullmatch(r"[\d\s.,%<>≤≥=+\-]+", item["text"])
            and re.search(r"\d", item["text"])
        ]
        if len(values) >= 3 and len(numeric_values) / len(values) >= 0.65:
            numeric_region_ids.update(item["id"] for item in column)
    return numeric_region_ids


def _recognize_tesseract_numeric_region(
    tesseract: Any,
    image: Image.Image,
    region: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if region is None:
        left, top, right, bottom = 0, 0, image.width, image.height
    else:
        points = region.get("box", [])
        if not points:
            return []
        left = min(float(point[0]) for point in points)
        top = min(float(point[1]) for point in points)
        right = max(float(point[0]) for point in points)
        bottom = max(float(point[1]) for point in points)

    text_height = max(1.0, bottom - top)
    padding = max(2, round(text_height * 0.3))
    crop_box = (
        max(0, round(left) - padding),
        max(0, round(top) - padding),
        min(image.width, round(right) + padding),
        min(image.height, round(bottom) + padding),
    )
    crop = image.crop(crop_box)
    scale = 2.0
    crop = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        Image.Resampling.LANCZOS,
    )
    result = tesseract.image_to_data(
        crop,
        lang="eng",
        config=(
            "--oem 1 --psm 7 "
            "-c tessedit_char_whitelist=0123456789.,%<>≤≥=+-"
        ),
        output_type=tesseract.Output.DICT,
    )
    lines = _parse_tesseract_result(result)
    for line in lines:
        for point in line["box"]:
            point[0] = point[0] / scale + crop_box[0]
            point[1] = point[1] / scale + crop_box[1]
    return lines


def _order_table_text_regions(
    text_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(text_regions) < 2:
        return text_regions

    def bounds(region: dict[str, Any]) -> tuple[float, float, float, float]:
        points = region.get("box", [])
        if not points:
            return 0.0, 0.0, 0.0, 0.0
        return (
            min(float(point[0]) for point in points),
            min(float(point[1]) for point in points),
            max(float(point[0]) for point in points),
            max(float(point[1]) for point in points),
        )

    valid_regions = [
        region
        for region in text_regions
        if bounds(region)[2] > bounds(region)[0]
        and bounds(region)[3] > bounds(region)[1]
    ]
    if len(valid_regions) < 2:
        return text_regions

    heights = sorted(
        bottom - top for _, top, _, bottom in map(bounds, valid_regions)
    )
    row_tolerance = max(2.0, heights[len(heights) // 2] * 0.55)
    rows: list[list[dict[str, Any]]] = []
    row_centers: list[float] = []
    for region in sorted(
        valid_regions,
        key=lambda item: (
            (bounds(item)[1] + bounds(item)[3]) / 2,
            bounds(item)[0],
        ),
    ):
        _, top, _, bottom = bounds(region)
        center = (top + bottom) / 2
        closest_row = min(
            range(len(rows)),
            key=lambda index: abs(row_centers[index] - center),
            default=None,
        )
        if (
            closest_row is None
            or abs(row_centers[closest_row] - center) > row_tolerance
        ):
            rows.append([region])
            row_centers.append(center)
            continue

        rows[closest_row].append(region)
        row_centers[closest_row] = sum(
            (bounds(item)[1] + bounds(item)[3]) / 2
            for item in rows[closest_row]
        ) / len(rows[closest_row])

    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: bounds(item)[0]))
    return ordered


def _recognize_tesseract_layout_ppocr_regions(
    tesseract: Any,
    image: Image.Image,
    regions: list[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    lines = []
    for x, y, width, height in regions:
        padding = max(2, round(min(width, height) * 0.06))
        crop_box = (
            max(0, x - padding),
            max(0, y - padding),
            min(image.width, x + width + padding),
            min(image.height, y + height + padding),
        )
        crop = image.crop(crop_box)
        crop, dewarp_offsets = _dewarp_page(crop)
        dewarped_size = crop.size
        text_regions = _detect_text_regions(crop)
        if text_regions:
            region_lines = _recognize_tesseract_text_regions(
                tesseract,
                crop,
                _order_text_regions(text_regions, []),
            )
        else:
            logger.warning(
                "PP-OCR không tìm thấy dòng trong khung DocLayout (%d, %d, %d, %d); "
                "dùng Tesseract trực tiếp trong khung.",
                x,
                y,
                width,
                height,
            )
            text_height = max(8.0, min(width, height) * 0.06)
            processed_crop, scale = _preprocess_tesseract_region(
                crop, text_height
            )
            if processed_crop is None:
                continue
            region_lines = _recognize_tesseract_page(
                tesseract, processed_crop, page_segmentation_mode=6
            )
            for line in region_lines:
                for point in line["box"]:
                    point[0] /= scale
                    point[1] /= scale

        if dewarp_offsets is not None:
            _map_lines_to_original(
                region_lines,
                PageTransform(np.eye(3, dtype=np.float32), dewarp_offsets),
                dewarped_size,
            )
        for line in region_lines:
            for point in line["box"]:
                point[0] += crop_box[0]
                point[1] += crop_box[1]
            lines.append(line)
    return _order_contact_label_values(lines)


def _order_text_regions(
    text_regions: list[dict[str, Any]],
    regions: list[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    column_ordered, found_columns = _order_detected_column_blocks(text_regions)
    if found_columns:
        return column_ordered
    if not regions:
        return _order_reading_columns(text_regions)

    grouped_regions: dict[int, list[dict[str, Any]]] = {}
    unassigned = []
    for text_region in text_regions:
        layout_index = _best_layout_region(text_region, regions)
        if layout_index is None:
            unassigned.append(text_region)
        else:
            grouped_regions.setdefault(layout_index, []).append(text_region)

    layout_groups = []
    for layout_index, (x, y, width, height) in enumerate(regions):
        if layout_index not in grouped_regions:
            continue
        layout_groups.append(
            {
                "box": [
                    [x, y],
                    [x + width, y],
                    [x + width, y + height],
                    [x, y + height],
                ],
                "layout_index": layout_index,
            }
        )
    layout_groups.extend(unassigned)

    ordered = []
    for group in _order_reading_columns(
        layout_groups, allow_single_item_columns=True
    ):
        layout_index = group.get("layout_index")
        if layout_index is None:
            ordered.append(group)
        else:
            ordered.extend(
                _order_reading_columns(grouped_regions[layout_index])
            )
    return ordered


def _order_detected_column_blocks(
    text_regions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    if len(text_regions) < 4:
        return text_regions, False

    def bounds(region: dict[str, Any]) -> tuple[float, float, float, float]:
        points = region.get("box", [])
        if not points:
            return 0.0, 0.0, 0.0, 0.0
        return (
            min(float(point[0]) for point in points),
            min(float(point[1]) for point in points),
            max(float(point[0]) for point in points),
            max(float(point[1]) for point in points),
        )

    heights = sorted(
        bottom - top
        for left, top, right, bottom in map(bounds, text_regions)
        if bottom > top and right > left
    )
    if not heights:
        return text_regions, False
    median_height = heights[len(heights) // 2]
    minimum_gap = max(1.0, median_height * 0.5)

    horizontal_gaps = []
    for first_index, first_region in enumerate(text_regions):
        first_left, first_top, first_right, first_bottom = bounds(first_region)
        first_height = first_bottom - first_top
        if first_height <= 0:
            continue
        for second_index, second_region in enumerate(text_regions):
            if first_index == second_index:
                continue
            second_left, second_top, second_right, second_bottom = bounds(
                second_region
            )
            second_height = second_bottom - second_top
            if second_height <= 0:
                continue
            vertical_overlap = max(
                0.0,
                min(first_bottom, second_bottom) - max(first_top, second_top),
            )
            overlap_ratio = vertical_overlap / min(first_height, second_height)
            if overlap_ratio < 0.5:
                continue
            if first_right + minimum_gap <= second_left:
                horizontal_gaps.append(
                    (
                        first_index,
                        second_index,
                        first_right,
                        second_left,
                        (first_top + first_bottom) / 2,
                    )
                )
            elif second_right + minimum_gap <= first_left:
                horizontal_gaps.append(
                    (
                        second_index,
                        first_index,
                        second_right,
                        first_left,
                        (first_top + first_bottom) / 2,
                    )
                )

    if len(horizontal_gaps) < 2:
        return text_regions, False

    candidates = sorted(
        (gap_left + gap_right) / 2
        for _, _, gap_left, gap_right, _ in horizontal_gaps
    )
    best_block = None
    best_support = 1
    for separator in candidates:
        matching_gaps = [
            gap
            for gap in horizontal_gaps
            if gap[2] < separator < gap[3]
        ]
        matching_gaps.sort(key=lambda gap: gap[4])
        gap_runs = []
        for gap in matching_gaps:
            if (
                not gap_runs
                or gap[4] - gap_runs[-1][-1][4] > median_height * 2.5
            ):
                gap_runs.append([gap])
            else:
                gap_runs[-1].append(gap)
        for run in gap_runs:
            row_centers = sorted(gap[4] for gap in run)
            distinct_rows = []
            for center in row_centers:
                if (
                    not distinct_rows
                    or center - distinct_rows[-1] >= median_height * 0.5
                ):
                    distinct_rows.append(center)
            support = len(distinct_rows)
            if support > best_support:
                best_support = support
                best_block = (separator, run)

    if best_block is None:
        return text_regions, False
    best_separator, run = best_block
    column_indexes = {
        index
        for gap in run
        for index in gap[:2]
    }
    column_top = min(bounds(text_regions[index])[1] for index in column_indexes)
    column_bottom = max(
        bounds(text_regions[index])[3] for index in column_indexes
    )
    extension_gap = max(2.0, median_height * 0.45)
    left_indexes = set()
    right_indexes = set()
    for index, region in enumerate(text_regions):
        if index in column_indexes:
            continue
        left, top, right, bottom = bounds(region)
        if right <= best_separator:
            left_indexes.add(index)
        elif left >= best_separator:
            right_indexes.add(index)

    while True:
        added = False
        for index in left_indexes | right_indexes:
            if index in column_indexes:
                continue
            _, top, _, bottom = bounds(text_regions[index])
            vertical_gap = max(column_top - bottom, top - column_bottom, 0.0)
            if vertical_gap <= extension_gap:
                column_indexes.add(index)
                column_top = min(column_top, top)
                column_bottom = max(column_bottom, bottom)
                added = True
        if not added:
            break

    left_column = sorted(
        (text_regions[index] for index in column_indexes if bounds(text_regions[index])[2] <= best_separator),
        key=lambda region: bounds(region)[1],
    )
    right_column = sorted(
        (text_regions[index] for index in column_indexes if bounds(text_regions[index])[0] >= best_separator),
        key=lambda region: bounds(region)[1],
    )
    if not left_column or not right_column:
        return text_regions, False

    left_text = [
        str(text_regions[index].get("text", "")).strip()
        for index in column_indexes
        if bounds(text_regions[index])[2] <= best_separator
    ]
    contact_labels = {"address", "phone", "email", "linkedin"}
    label_count = sum(
        text.strip().casefold().rstrip(":：") in contact_labels
        or text.strip().endswith((":", "："))
        for text in left_text
    )
    label_column = label_count >= max(2, len(left_text) // 2)
    if label_column:
        block = _order_reading_rows(left_column + right_column)
    else:
        block = left_column + right_column

    remaining = [
        region
        for index, region in enumerate(text_regions)
        if index not in column_indexes
    ]
    before = [region for region in remaining if bounds(region)[1] < column_top]
    after = [region for region in remaining if bounds(region)[1] >= column_top]
    before_ordered, _ = _order_detected_column_blocks(before)
    after_ordered, _ = _order_detected_column_blocks(after)
    return (
        _order_reading_rows(before_ordered)
        + block
        + _order_reading_rows(after_ordered),
        True,
    )


def _order_reading_columns(
    text_regions: list[dict[str, Any]],
    allow_single_item_columns: bool = False,
) -> list[dict[str, Any]]:
    def bounds(region: dict[str, Any]) -> tuple[float, float, float, float]:
        points = region["box"]
        return (
            min(float(point[0]) for point in points),
            min(float(point[1]) for point in points),
            max(float(point[0]) for point in points),
            max(float(point[1]) for point in points),
        )

    if len(text_regions) < 2 or (
        len(text_regions) < 4 and not allow_single_item_columns
    ):
        return _order_reading_rows(text_regions)

    ordered_by_x = sorted(text_regions, key=lambda region: bounds(region)[0])
    heights = sorted(bottom - top for _, top, _, bottom in map(bounds, text_regions))
    minimum_gap = max(1.0, heights[len(heights) // 2] * 0.5)
    best_split = None
    best_gap = minimum_gap

    for split_index in range(1, len(ordered_by_x)):
        left_column = ordered_by_x[:split_index]
        right_column = ordered_by_x[split_index:]
        left_bounds = [bounds(region) for region in left_column]
        right_bounds = [bounds(region) for region in right_column]
        gap = min(item[0] for item in right_bounds) - max(
            item[2] for item in left_bounds
        )
        if gap <= best_gap:
            continue

        left_top = min(item[1] for item in left_bounds)
        left_bottom = max(item[3] for item in left_bounds)
        right_top = min(item[1] for item in right_bounds)
        right_bottom = max(item[3] for item in right_bounds)
        vertical_overlap = max(
            0.0, min(left_bottom, right_bottom) - max(left_top, right_top)
        )
        overlap_ratio = vertical_overlap / min(
            max(1.0, left_bottom - left_top),
            max(1.0, right_bottom - right_top),
        )
        if overlap_ratio < 0.5:
            continue
        best_gap = gap
        best_split = split_index

    if best_split is None:
        return _order_reading_rows(text_regions)

    return _order_reading_columns(
        ordered_by_x[:best_split], allow_single_item_columns
    ) + _order_reading_columns(
        ordered_by_x[best_split:], allow_single_item_columns
    )


def _order_reading_rows(
    text_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def bounds(region: dict[str, Any]) -> tuple[float, float, float, float]:
        points = region["box"]
        return (
            min(float(point[0]) for point in points),
            min(float(point[1]) for point in points),
            max(float(point[0]) for point in points),
            max(float(point[1]) for point in points),
        )

    rows: list[list[dict[str, Any]]] = []
    for region in sorted(text_regions, key=lambda item: bounds(item)[1]):
        left, top, right, bottom = bounds(region)
        height = max(1.0, bottom - top)
        matching_row = None
        best_overlap = 0.0
        for row in rows:
            row_bounds = [bounds(item) for item in row]
            row_top = min(item[1] for item in row_bounds)
            row_bottom = max(item[3] for item in row_bounds)
            overlap = max(0.0, min(bottom, row_bottom) - max(top, row_top))
            overlap_ratio = overlap / min(height, max(1.0, row_bottom - row_top))
            if overlap_ratio >= 0.5 and overlap_ratio > best_overlap:
                matching_row = row
                best_overlap = overlap_ratio
        if matching_row is None:
            rows.append([region])
        else:
            matching_row.append(region)

    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: bounds(item)[0]))
    return ordered


def _best_layout_region(
    line: dict[str, Any],
    regions: list[tuple[int, int, int, int]],
) -> int | None:
    points = line.get("box")
    if not points:
        return None

    left = min(float(point[0]) for point in points)
    top = min(float(point[1]) for point in points)
    right = max(float(point[0]) for point in points)
    bottom = max(float(point[1]) for point in points)
    line_area = max(0.0, right - left) * max(0.0, bottom - top)
    if not line_area:
        return None

    best_region = None
    best_overlap = 0.0
    for index, (x, y, width, height) in enumerate(regions):
        overlap_width = max(0.0, min(right, x + width) - max(left, x))
        overlap_height = max(0.0, min(bottom, y + height) - max(top, y))
        overlap_ratio = overlap_width * overlap_height / line_area
        if overlap_ratio > best_overlap:
            best_region = index
            best_overlap = overlap_ratio
    return best_region if best_overlap >= 0.25 else None


def _recognize_tesseract_text_regions(
    tesseract: Any,
    image: Image.Image,
    text_regions: list[dict[str, Any]],
    order_contact_values: bool = True,
) -> list[dict[str, Any]]:
    lines = []
    for region_index, text_region in enumerate(text_regions):
        points = text_region["box"]
        left = min(float(point[0]) for point in points)
        top = min(float(point[1]) for point in points)
        right = max(float(point[0]) for point in points)
        bottom = max(float(point[1]) for point in points)
        text_height = max(1.0, bottom - top)
        horizontal_padding = max(2, round(text_height * 0.15))
        vertical_padding = _text_line_vertical_padding(
            text_regions, region_index, left, top, right, bottom
        )
        crop_box = (
            max(0, round(left) - horizontal_padding),
            max(0, round(top) - vertical_padding),
            min(image.width, round(right) + horizontal_padding),
            min(image.height, round(bottom) + vertical_padding),
        )
        crop = image.crop(crop_box)
        crop, scale = _preprocess_tesseract_region(crop, text_height)
        if crop is None:
            continue
        region_lines = _recognize_tesseract_page(
            tesseract, crop, page_segmentation_mode=7
        )
        for line in region_lines:
            for point in line["box"]:
                point[0] = point[0] / scale + crop_box[0]
                point[1] = point[1] / scale + crop_box[1]
            if text_region["confidence"] is not None:
                line["detection_confidence"] = text_region["confidence"]
            lines.append(line)
    return _order_contact_label_values(lines) if order_contact_values else lines


def _text_line_vertical_padding(
    text_regions: list[dict[str, Any]],
    region_index: int,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> int:
    max_padding = 3
    padding = max_padding
    for other_index, text_region in enumerate(text_regions):
        if other_index == region_index:
            continue
        points = text_region["box"]
        other_left = min(float(point[0]) for point in points)
        other_top = min(float(point[1]) for point in points)
        other_right = max(float(point[0]) for point in points)
        other_bottom = max(float(point[1]) for point in points)
        if min(right, other_right) <= max(left, other_left):
            continue
        if other_bottom <= top:
            gap = top - other_bottom
        elif other_top >= bottom:
            gap = other_top - bottom
        else:
            continue
        padding = min(padding, max(0, int((gap - 1) // 2)))
    return padding


def _order_contact_label_values(
    lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    contact_labels = {"address", "phone", "email", "linkedin"}

    def bounds(line: dict[str, Any]) -> tuple[float, float, float, float]:
        points = line.get("box", [])
        if not points:
            return 0.0, 0.0, 0.0, 0.0
        return (
            min(float(point[0]) for point in points),
            min(float(point[1]) for point in points),
            max(float(point[0]) for point in points),
            max(float(point[1]) for point in points),
        )

    label_indices = [
        index
        for index, line in enumerate(lines)
        if str(line.get("text", "")).strip().casefold().rstrip(":：")
        in contact_labels
    ]
    if len(label_indices) < 2:
        return lines

    pairs = []
    used_value_indices: set[int] = set()
    for label_index in sorted(
        label_indices, key=lambda index: bounds(lines[index])[1]
    ):
        _, label_top, label_right, label_bottom = bounds(lines[label_index])
        label_height = max(1.0, label_bottom - label_top)
        candidates = []
        for value_index, value_line in enumerate(lines):
            if value_index in label_indices or value_index in used_value_indices:
                continue
            value_left, value_top, _, value_bottom = bounds(value_line)
            value_height = max(1.0, value_bottom - value_top)
            if value_left < label_right - max(8.0, label_height * 0.25):
                continue
            vertical_overlap = max(
                0.0,
                min(label_bottom, value_bottom) - max(label_top, value_top),
            )
            overlap_ratio = vertical_overlap / min(label_height, value_height)
            center_distance = abs(
                (label_top + label_bottom) / 2
                - (value_top + value_bottom) / 2
            )
            if overlap_ratio < 0.3 and center_distance > max(
                label_height, value_height
            ) * 0.55:
                continue
            candidates.append(
                (center_distance, value_left - label_right, value_index)
            )
        if candidates:
            _, _, value_index = min(candidates)
            used_value_indices.add(value_index)
            pairs.append((label_index, value_index))

    if len(pairs) < 2:
        return lines

    ordered_pairs = sorted(pairs, key=lambda pair: bounds(lines[pair[0]])[1])
    paired_indices = {index for pair in ordered_pairs for index in pair}
    insertion_index = min(paired_indices)
    ordered_contacts = [
        line
        for label_index, value_index in ordered_pairs
        for line in (lines[label_index], lines[value_index])
    ]
    reordered = []
    for index, line in enumerate(lines):
        if index == insertion_index:
            reordered.extend(ordered_contacts)
        if index not in paired_indices:
            reordered.append(line)
    return reordered


@lru_cache(maxsize=1)
def _get_layout_detector() -> tuple[Any, str]:
    try:
        import torch
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            HfHubHTTPError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError as error:
        raise OCRUnavailableError(
            "Không import được DocLayout-YOLO trong môi trường "
            f"{sys.executable}: {error}. Hãy cài requirements.txt hoặc "
            "requirements-gpu.txt bằng chính Python của môi trường này."
        ) from error

    from app.database import DATA_DIR

    try:
        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained",
            filename="doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt",
            revision="51b2c54a642013022382c3d2bd76e84b07656b4a",
            cache_dir=str(DATA_DIR / "models"),
        )
    except (
        EntryNotFoundError,
        HfHubHTTPError,
        OSError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    ) as error:
        raise OCRUnavailableError(
            "Không tải được model DocLayout-YOLO. Kiểm tra kết nối Internet "
            "rồi thử lại."
        ) from error

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        return YOLOv10(model_path), device
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise OCRUnavailableError(
            f"Không khởi tạo được model DocLayout-YOLO: {error}"
        ) from error


def _recognize_tesseract_page(
    tesseract: Any, image: Image.Image, page_segmentation_mode: int = 6
) -> list[dict[str, Any]]:
    return _parse_tesseract_result(
        tesseract.image_to_data(
            image,
            lang="vie",
            config=f"--oem 1 --psm {page_segmentation_mode}",
            output_type=tesseract.Output.DICT,
        )
    )


def _preprocess_tesseract_region(
    image: Image.Image, text_height: float
) -> tuple[Image.Image | None, float]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    if contrast < 8:
        return None, 1.0
    if contrast < 45:
        gray = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8)).apply(gray)
        image = Image.fromarray(gray)

    scale = 2.0 if text_height < 10 else 1.5 if text_height < 18 else 1.0
    if scale > 1:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image, scale


@lru_cache(maxsize=1)
def _get_tesseract() -> Any:
    try:
        import pytesseract
    except ImportError as error:
        raise OCRUnavailableError(
            "Chưa cài pytesseract. Cài requirements.txt hoặc "
            "requirements-gpu.txt rồi khởi động lại."
        ) from error

    try:
        pytesseract.get_tesseract_version()
        languages = pytesseract.get_languages(config="")
    except pytesseract.TesseractNotFoundError as error:
        raise OCRUnavailableError(
            "Không tìm thấy Tesseract. Kích hoạt môi trường Conda của ứng dụng "
            "và cài Tesseract bằng: conda install -c conda-forge tesseract"
        ) from error
    except pytesseract.TesseractError as error:
        raise OCRUnavailableError(
            f"Không thể khởi tạo Tesseract hoặc dữ liệu ngôn ngữ: {error}"
        ) from error

    if "vie" not in languages:
        raise OCRUnavailableError(
            "Tesseract chưa có dữ liệu tiếng Việt (vie). Hãy cài lại gói "
            "tesseract từ conda-forge trong môi trường ứng dụng."
        )
    return pytesseract


@lru_cache(maxsize=1)
def _get_text_detector() -> Any:
    try:
        import paddle
        from paddleocr import TextDetection
    except ImportError as error:
        raise OCRUnavailableError(
            "PP-OCRv6 Medium cần PaddleOCR 3.7.0 và PaddlePaddle 3.2.0. Cài "
            "requirements.txt hoặc requirements-gpu.txt trong môi trường ứng "
            "dụng rồi khởi động lại."
        ) from error

    use_gpu = (
        paddle.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0
    )
    device = "gpu:0" if use_gpu else "cpu"
    try:
        return TextDetection(
            model_name="PP-OCRv6_medium_det",
            device=device,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise OCRUnavailableError(
            "Không tải hoặc khởi tạo được PP-OCRv6 Medium detector. Kiểm tra "
            f"PaddlePaddle, kết nối Internet và cache model: {error}"
        ) from error


def _align_page(image: Image.Image) -> tuple[Image.Image, PageTransform]:
    oriented, orientation_transform = _correct_page_orientation(image)
    rgb = np.asarray(oriented)
    height, width = rgb.shape[:2]
    transform = orientation_transform

    page = _find_page_quad(rgb)
    if page is not None:
        top_left, top_right, bottom_right, bottom_left = page
        output_width = max(
            int(round(np.linalg.norm(bottom_right - bottom_left))),
            int(round(np.linalg.norm(top_right - top_left))),
        )
        output_height = max(
            int(round(np.linalg.norm(top_right - bottom_right))),
            int(round(np.linalg.norm(top_left - bottom_left))),
        )
        output_width = max(1, output_width)
        output_height = max(1, output_height)
        target = np.asarray(
            [
                [0, 0],
                [output_width - 1, 0],
                [output_width - 1, output_height - 1],
                [0, output_height - 1],
            ],
            dtype=np.float32,
        )
        perspective_transform = cv2.getPerspectiveTransform(page, target)
        rgb = cv2.warpPerspective(
            rgb,
            perspective_transform,
            (output_width, output_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        transform = perspective_transform @ transform
    else:
        skew = _estimate_skew_angle(rgb)
        if (
            skew is not None
            and DESKEW_MIN_ANGLE <= abs(skew) <= DESKEW_MAX_ANGLE
        ):
            height, width = rgb.shape[:2]
            radians = np.deg2rad(skew)
            cosine, sine = abs(np.cos(radians)), abs(np.sin(radians))
            output_width = int(np.ceil(width * cosine + height * sine))
            output_height = int(np.ceil(height * cosine + width * sine))
            transform_2d = cv2.getRotationMatrix2D(
                (width / 2, height / 2), skew, 1.0
            )
            transform_2d[0, 2] += output_width / 2 - width / 2
            transform_2d[1, 2] += output_height / 2 - height / 2
            skew_transform = np.vstack([transform_2d, [0, 0, 1]]).astype(
                np.float32
            )
            rgb = cv2.warpAffine(
                rgb,
                transform_2d,
                (output_width, output_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            transform = skew_transform @ transform

    aligned_image = Image.fromarray(rgb)
    aligned_image, dewarp_offsets = _dewarp_page(aligned_image)
    if dewarp_offsets is not None:
        logger.info("Đã dewarp nhẹ toàn trang trước khi phát hiện bố cục.")
    return aligned_image, PageTransform(transform, dewarp_offsets)


def _correct_page_orientation(
    image: Image.Image,
) -> tuple[Image.Image, np.ndarray]:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    try:
        import pytesseract

        orientation = pytesseract.image_to_osd(image, config="--psm 0")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        logger.debug("Không xác định được hướng trang bằng Tesseract OSD: %s", error)
        return image, np.eye(3, dtype=np.float32)
    except pytesseract.TesseractError as error:
        logger.debug("Tesseract OSD không xác định được hướng trang: %s", error)
        return image, np.eye(3, dtype=np.float32)

    rotation_match = re.search(r"^\s*Rotate:\s*(\d+)\s*$", orientation, re.MULTILINE)
    confidence_match = re.search(
        r"^\s*Orientation confidence:\s*([\d.]+)\s*$",
        orientation,
        re.MULTILINE,
    )
    if rotation_match is None or confidence_match is None:
        return image, np.eye(3, dtype=np.float32)

    rotation = int(rotation_match.group(1)) % 360
    confidence = float(confidence_match.group(1))
    if rotation not in {90, 180, 270} or confidence < ORIENTATION_MIN_CONFIDENCE:
        return image, np.eye(3, dtype=np.float32)

    if rotation == 90:
        corrected = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        transform = np.asarray(
            [[0, -1, height - 1], [1, 0, 0], [0, 0, 1]],
            dtype=np.float32,
        )
    elif rotation == 180:
        corrected = cv2.rotate(rgb, cv2.ROTATE_180)
        transform = np.asarray(
            [[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]],
            dtype=np.float32,
        )
    else:
        corrected = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        transform = np.asarray(
            [[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]],
            dtype=np.float32,
        )

    logger.info(
        "Tesseract OSD tự xoay trang %d° (độ tin cậy %.1f).",
        rotation,
        confidence,
    )
    return Image.fromarray(corrected), transform


def _dewarp_page(image: Image.Image) -> tuple[Image.Image, np.ndarray | None]:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if min(height, width) < DEWARP_MIN_PAGE_DIMENSION:
        return image, None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    threshold_block_size = min(
        DEWARP_THRESHOLD_BLOCK_SIZE,
        min(height, width) // 2 * 2 - 1,
    )
    if threshold_block_size < DEWARP_MIN_THRESHOLD_BLOCK_SIZE:
        return image, None
    ink = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        threshold_block_size,
        DEWARP_THRESHOLD_C,
    )

    projection = np.convolve(
        ink.sum(axis=1).astype(np.float32),
        np.ones(DEWARP_PROJECTION_KERNEL_SIZE, dtype=np.float32)
        / DEWARP_PROJECTION_KERNEL_SIZE,
        mode="same",
    )
    active_rows = projection > max(
        width * DEWARP_MIN_INK_WIDTH_RATIO,
        float(projection.max()) * DEWARP_PEAK_PROJECTION_RATIO,
    )
    bands = []
    row = 0
    while row < height:
        if not active_rows[row]:
            row += 1
            continue
        start = row
        while row < height and active_rows[row]:
            row += 1
        end = row
        if DEWARP_MIN_LINE_BAND_HEIGHT <= end - start <= max(
            DEWARP_MIN_MAX_LINE_BAND_HEIGHT,
            round(height * DEWARP_MAX_LINE_BAND_HEIGHT_RATIO),
        ):
            bands.append((start, end))
    if len(bands) < DEWARP_MIN_TEXT_LINES:
        return image, None

    strip_width = max(1, int(np.ceil(width / DEWARP_STRIP_COUNT)))
    strip_centers = np.arange(DEWARP_STRIP_COUNT, dtype=np.float32) * strip_width
    displacements = np.full(DEWARP_STRIP_COUNT, np.nan, dtype=np.float32)
    for strip_index, left in enumerate(strip_centers.astype(int)):
        right = min(width, left + strip_width)
        if right <= left:
            continue
        line_displacements = []
        for top, bottom in bands:
            margin = max(2, (bottom - top) // 2)
            region_top = max(0, top - margin)
            region_bottom = min(height, bottom + margin)
            local_projection = ink[region_top:region_bottom, left:right].sum(axis=1)
            if (
                local_projection.max(initial=0)
                < (right - left) * 255 * DEWARP_STRIP_INK_RATIO
            ):
                continue
            local_rows = np.arange(region_top, region_bottom)
            local_center = float(
                np.average(local_rows, weights=local_projection.astype(np.float64))
            )
            band_center = (top + bottom) / 2
            line_displacements.append(local_center - band_center)
        if len(line_displacements) >= DEWARP_MIN_TEXT_LINES // 2:
            displacements[strip_index] = float(np.median(line_displacements))

    valid = np.isfinite(displacements)
    if valid.sum() < max(
        DEWARP_MIN_VALID_STRIPS,
        round(DEWARP_STRIP_COUNT * DEWARP_MIN_VALID_STRIP_RATIO),
    ):
        return image, None
    displacements = np.interp(
        np.arange(DEWARP_STRIP_COUNT),
        np.flatnonzero(valid),
        displacements[valid],
    ).astype(np.float32)
    strip_x = np.minimum(
        strip_centers + strip_width / 2,
        width - 1,
    )
    slope, intercept = np.polyfit(strip_x, displacements, 1)
    displacements -= slope * strip_x + intercept

    smoothing_width = max(
        DEWARP_MIN_SMOOTHING_WIDTH,
        int(round(DEWARP_STRIP_COUNT * DEWARP_SMOOTHING_RATIO)),
    )
    if smoothing_width % 2 == 0:
        smoothing_width += 1
    displacements = cv2.GaussianBlur(
        displacements.reshape(1, -1),
        (smoothing_width, 1),
        0,
    ).reshape(-1)

    maximum_displacement = height * DEWARP_MAX_DISPLACEMENT_RATIO
    displacements = np.clip(
        displacements,
        -maximum_displacement,
        maximum_displacement,
    )
    if (
        float(np.ptp(displacements))
        < height * DEWARP_MIN_DISPLACEMENT_RATIO
    ):
        return image, None

    map_x = np.broadcast_to(
        np.arange(width, dtype=np.float32), (height, width)
    ).copy()
    map_y = np.broadcast_to(
        np.arange(height, dtype=np.float32)[:, None], (height, width)
    ).copy()
    per_pixel_displacement = np.interp(
        np.arange(width),
        strip_x,
        displacements,
    ).astype(np.float32)
    map_y += per_pixel_displacement[None, :]
    corrected = cv2.remap(
        rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(corrected), per_pixel_displacement


def _find_page_quad(rgb: np.ndarray) -> np.ndarray | None:
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, max(lower + 1, 1.33 * median)))
    edges = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = float(width * height)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < image_area * 0.20:
            break
        if area > image_area * 0.995:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        points = _order_quad(approx.reshape(4, 2).astype(np.float32))
        top_left, top_right, bottom_right, bottom_left = points
        quad_width = max(
            np.linalg.norm(top_right - top_left),
            np.linalg.norm(bottom_right - bottom_left),
        )
        quad_height = max(
            np.linalg.norm(bottom_left - top_left),
            np.linalg.norm(bottom_right - top_right),
        )
        aspect = quad_width / max(quad_height, 1.0)
        if 0.25 <= aspect <= 4.0:
            return points
    max_dimension = 1200
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        small_gray = cv2.resize(
            gray,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small_gray = gray
    if int(small_gray.max()) - int(small_gray.min()) < 32:
        return None
    _, paper_mask = cv2.threshold(
        small_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    close_size = max(5, round(min(paper_mask.shape) * 0.015))
    if close_size % 2 == 0:
        close_size += 1
    paper_mask = cv2.morphologyEx(
        paper_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)
        ),
        iterations=2,
    )
    contours, _ = cv2.findContours(
        paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    small_area = float(paper_mask.shape[0] * paper_mask.shape[1])
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < small_area * 0.20:
            break
        if area > small_area * 0.98:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        points = _order_quad(approx.reshape(4, 2).astype(np.float32))
        points /= scale
        top_left, top_right, bottom_right, bottom_left = points
        quad_width = max(
            np.linalg.norm(top_right - top_left),
            np.linalg.norm(bottom_right - bottom_left),
        )
        quad_height = max(
            np.linalg.norm(bottom_left - top_left),
            np.linalg.norm(bottom_right - top_right),
        )
        aspect = quad_width / max(quad_height, 1.0)
        if 0.25 <= aspect <= 4.0:
            return points
    return None


def _order_quad(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)


def _estimate_skew_angle(rgb: np.ndarray) -> float | None:
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_length = max(40, int(min(height, width) * 0.15))
    threshold = max(30, int(min(height, width) * 0.08))
    segments = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=max(5, int(min(height, width) * 0.02)),
    )
    if segments is None:
        return None

    candidates: list[tuple[float, float]] = []
    for segment in segments[:, 0]:
        x1, y1, x2, y2 = (float(value) for value in segment)
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        if abs(angle) <= DESKEW_MAX_ANGLE:
            candidates.append((angle, length))

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate[0])
    half_weight = sum(weight for _, weight in candidates) / 2
    accumulated = 0.0
    for angle, weight in candidates:
        accumulated += weight
        if accumulated >= half_weight:
            return angle
    return None


def _map_lines_to_original(
    lines: list[dict[str, Any]],
    image_to_aligned: np.ndarray | PageTransform,
    image_size: tuple[int, int],
) -> None:
    if isinstance(image_to_aligned, PageTransform):
        if image_to_aligned.dewarp_offsets is not None:
            x_coordinates = np.arange(len(image_to_aligned.dewarp_offsets))
            for line in lines:
                points = np.asarray(line["box"], dtype=np.float32)
                points[:, 1] += np.interp(
                    points[:, 0],
                    x_coordinates,
                    image_to_aligned.dewarp_offsets,
                )
                line["box"] = points.tolist()
        transform = image_to_aligned.matrix
    else:
        transform = image_to_aligned
    aligned_to_original = np.linalg.inv(transform)
    width, height = image_size
    for line in lines:
        points = np.asarray(line["box"], dtype=np.float32).reshape(-1, 1, 2)
        original_points = cv2.perspectiveTransform(points, aligned_to_original)
        original_points[:, 0, 0] = np.clip(original_points[:, 0, 0], 0, width - 1)
        original_points[:, 0, 1] = np.clip(original_points[:, 0, 1], 0, height - 1)
        line["box"] = original_points[:, 0, :].tolist()


def _parse_tesseract_result(result: dict[str, list[Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, int], dict[str, Any]] = {}
    for index, raw_text in enumerate(result["text"]):
        text = _normalize_text(str(raw_text)).strip()
        if not text:
            continue

        key = tuple(
            int(result[column][index])
            for column in ("block_num", "par_num", "line_num")
        )
        left = int(result["left"][index])
        top = int(result["top"][index])
        right = left + int(result["width"][index])
        bottom = top + int(result["height"][index])
        group = groups.setdefault(
            key,
            {
                "words": [],
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "confidences": [],
            },
        )
        group["words"].append(text)
        group["left"] = min(group["left"], left)
        group["top"] = min(group["top"], top)
        group["right"] = max(group["right"], right)
        group["bottom"] = max(group["bottom"], bottom)
        confidence = float(result["conf"][index])
        if confidence >= 0:
            group["confidences"].append(confidence)

    lines = []
    for group in groups.values():
        left, top, right, bottom = (
            group["left"],
            group["top"],
            group["right"],
            group["bottom"],
        )
        confidences = group["confidences"]
        lines.append(
            {
                "box": [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                ],
                "text": " ".join(group["words"]),
                "confidence": (
                    sum(confidences) / len(confidences) / 100
                    if confidences
                    else None
                ),
            }
        )
    return lines


def _join_lines(lines: list[dict[str, Any]]) -> str:
    return _normalize_text("\n".join(line["text"] for line in lines))


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)
