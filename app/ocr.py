import logging
import sys
import unicodedata
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


class OCRUnavailableError(RuntimeError):
    pass


logger = logging.getLogger(__name__)


def recognize(image_path: str, engine: str) -> tuple[str, list[dict[str, Any]]]:
    if engine not in {"ppocrv6", "paddle", "tesseract"}:
        raise ValueError(f"OCR engine không hợp lệ: {engine}")

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    aligned_image, image_to_aligned = _align_page(image)

    if engine in {"ppocrv6", "paddle"}:
        tesseract = _get_tesseract()
        layout_regions = []
        try:
            _get_layout_detector()
            layout_regions = _detect_layout_regions(aligned_image)
        except (OCRUnavailableError, OSError, RuntimeError, ValueError) as error:
            logger.warning(
                "Không chạy được DocLayout-YOLO; sẽ sắp dòng PaddleOCR theo "
                "tọa độ: %s",
                error,
            )
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
    return _layout_regions_from_detections(detections, image.size)


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
    boxes = [(x, y, width, height) for x, y, width, height, _ in regions]
    return sorted(boxes, key=lambda region: (region[1], region[0]))


def _order_text_regions(
    text_regions: list[dict[str, Any]],
    regions: list[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    if not regions:
        return _order_reading_rows(text_regions)

    grouped_regions: dict[int, list[dict[str, Any]]] = {}
    unassigned = []
    for text_region in text_regions:
        layout_index = _best_layout_region(text_region, regions)
        if layout_index is None:
            unassigned.append(text_region)
        else:
            grouped_regions.setdefault(layout_index, []).append(text_region)

    ordered = []
    for layout_index in sorted(grouped_regions):
        ordered.extend(_order_reading_rows(grouped_regions[layout_index]))
    ordered.extend(_order_reading_rows(unassigned))
    return ordered


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
) -> list[dict[str, Any]]:
    lines = []
    for text_region in text_regions:
        points = text_region["box"]
        left = min(float(point[0]) for point in points)
        top = min(float(point[1]) for point in points)
        right = max(float(point[0]) for point in points)
        bottom = max(float(point[1]) for point in points)
        text_height = max(1.0, bottom - top)
        horizontal_padding = max(2, round(text_height * 0.15))
        crop_box = (
            max(0, round(left) - horizontal_padding),
            max(0, round(top)),
            min(image.width, round(right) + horizontal_padding),
            min(image.height, round(bottom)),
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
    return lines


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


def _align_page(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
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
        transform = cv2.getPerspectiveTransform(page, target)
        aligned = cv2.warpPerspective(
            rgb,
            transform,
            (output_width, output_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(aligned), transform

    skew = _estimate_skew_angle(rgb)
    if skew is None or abs(skew) < 0.4:
        return image, np.eye(3, dtype=np.float32)

    radians = np.deg2rad(skew)
    cosine, sine = abs(np.cos(radians)), abs(np.sin(radians))
    output_width = int(np.ceil(width * cosine + height * sine))
    output_height = int(np.ceil(height * cosine + width * sine))
    transform_2d = cv2.getRotationMatrix2D((width / 2, height / 2), skew, 1.0)
    transform_2d[0, 2] += output_width / 2 - width / 2
    transform_2d[1, 2] += output_height / 2 - height / 2
    transform = np.vstack([transform_2d, [0, 0, 1]]).astype(np.float32)
    aligned = cv2.warpAffine(
        rgb,
        transform_2d,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(aligned), transform


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
        if abs(angle) <= 15:
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
    lines: list[dict[str, Any]], image_to_aligned: np.ndarray, image_size: tuple[int, int]
) -> None:
    aligned_to_original = np.linalg.inv(image_to_aligned)
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
