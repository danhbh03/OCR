import unicodedata
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


class OCRUnavailableError(RuntimeError):
    pass


def recognize(image_path: str, engine: str) -> tuple[str, list[dict[str, Any]]]:
    if engine not in {"paddle", "vietocr"}:
        raise ValueError(f"OCR engine không hợp lệ: {engine}")

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    aligned_image, image_to_aligned = _align_page(image)
    processed_image = _preprocess_image(aligned_image)
    image_array = cv2.cvtColor(np.asarray(processed_image), cv2.COLOR_RGB2BGR)
    predictor = None
    if engine == "vietocr":
        try:
            predictor = _get_vietocr()
        except ImportError as error:
            raise OCRUnavailableError(
                "Không tải hoặc khởi tạo được VietOCR/PyTorch "
                f"({error}). Kiểm tra requirements-ocr-viet.txt, "
                "requirements-torch-cpu.txt và setuptools<81."
            ) from error

    # Load PyTorch first: Paddle and PyTorch can conflict over DLLs on Windows.
    try:
        detected_lines = _get_paddle().ocr(image_array, cls=True)
    except ImportError as error:
        raise OCRUnavailableError(
            "Chưa cài PaddleOCR. Cài requirements-ocr-paddle.txt rồi khởi động lại."
        ) from error

    lines = _parse_paddle_result(detected_lines)
    if engine == "paddle":
        _map_lines_to_original(lines, image_to_aligned, image.size)
        return _join_lines(lines), lines

    if predictor is None:
        raise RuntimeError("VietOCR predictor was not initialized.")

    recognized_lines: list[dict[str, Any]] = []
    for line in lines:
        crop = _rectify_line(processed_image, line["box"])
        line["text"] = _normalize_text(predictor.predict(crop))
        line["confidence"] = None
        recognized_lines.append(line)
    _map_lines_to_original(recognized_lines, image_to_aligned, image.size)
    return _join_lines(recognized_lines), recognized_lines


@lru_cache(maxsize=1)
def _get_paddle() -> Any:
    from paddleocr import PaddleOCR
    import paddle

    use_gpu = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0

    return PaddleOCR(use_angle_cls=True, lang="vi", use_gpu=use_gpu)


@lru_cache(maxsize=1)
def _get_vietocr() -> Any:
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor
    import torch

    config = Cfg.load_config_from_name("vgg_transformer")
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["predictor"]["beamsearch"] = False
    return Predictor(config)


def _preprocess_image(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(
        lightness
    )
    enhanced = cv2.merge((lightness, a_channel, b_channel))
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB))


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


def _rectify_line(image: Image.Image, box: list[list[float]]) -> Image.Image:
    points = _order_quad(np.asarray(box, dtype=np.float32))

    top_left, top_right, bottom_right, bottom_left = points
    width = max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
    )
    height = max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right),
    )
    target_width = max(1, int(round(width)))
    target_height = max(1, int(round(height)))
    target = np.asarray(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, target)
    rectified = cv2.warpPerspective(
        np.asarray(image),
        transform,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(rectified)


def _parse_paddle_result(result: Any) -> list[dict[str, Any]]:
    if not result or not result[0]:
        return []

    lines = []
    for item in result[0]:
        if not item or len(item) < 2:
            continue
        box, recognition = item
        text, confidence = recognition
        lines.append(
            {
                "box": [[float(point[0]), float(point[1])] for point in box],
                "text": _normalize_text(str(text)),
                "confidence": float(confidence),
            }
        )
    return lines


def _join_lines(lines: list[dict[str, Any]]) -> str:
    return _normalize_text("\n".join(line["text"] for line in lines))


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)
