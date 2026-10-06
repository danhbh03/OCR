import os
import hashlib
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app import database
from app.corrections import extract_corrected_samples
from app.ocr import OCRUnavailableError, recognize


MAX_UPLOAD_BYTES = int(os.getenv("OCR_MAX_UPLOAD_BYTES", 15 * 1024 * 1024))
ALLOWED_FORMATS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "TIFF": ".tiff",
}
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.initialize_database()
    yield


app = FastAPI(title="OCR tiếng Việt local", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Correction(BaseModel):
    edited_text: str


class RegionSelection(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    edited_text: str | None = None


class DetectionAdjustment(RegionSelection):
    detection_index: int = Field(ge=0)
    edited_line_index: int | None = Field(default=None, ge=0)


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/documents")
async def documents() -> list[dict]:
    return database.list_documents()


@app.get("/api/training-samples")
async def training_samples(engine: str | None = None) -> list[dict[str, Any]]:
    if engine not in {None, "paddle", "vietocr"}:
        raise HTTPException(
            status_code=422, detail="Engine phải là paddle hoặc vietocr."
        )
    samples = database.list_training_samples()
    return [sample for sample in samples if engine is None or sample["engine"] == engine]


@app.get("/api/training-samples/{sample_id}/image")
async def training_sample_image(sample_id: str) -> FileResponse:
    sample = database.get_training_sample(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy mẫu huấn luyện.")
    image_path = (database.DATA_DIR / sample["image_path"]).resolve()
    if image_path.parent != (database.DATA_DIR / "training_samples").resolve():
        raise HTTPException(status_code=404, detail="Đường dẫn mẫu không hợp lệ.")
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh mẫu.")
    return FileResponse(image_path)


@app.get("/api/detection-samples")
async def detection_samples() -> list[dict[str, Any]]:
    return [
        {
            **sample,
            "task": "detection",
            "target_engine": "paddle-det",
        }
        for sample in database.list_detection_samples()
    ]


@app.get("/api/training-data/paddle/detection")
async def paddle_detection_training_data() -> list[dict[str, Any]]:
    return [
        {
            "image_url": sample["image_url"],
            "label": [
                {
                    "transcription": sample["text"],
                    "points": sample["points"],
                }
            ],
            "target_engine": "paddle-det",
        }
        for sample in database.list_detection_samples()
    ]


@app.get("/api/detection-samples/{sample_id}/image")
async def detection_sample_image(sample_id: str) -> FileResponse:
    sample = database.get_detection_sample(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy mẫu phát hiện.")
    image_path = (database.DATA_DIR / sample["image_path"]).resolve()
    if image_path.parent != (database.DATA_DIR / "detection_samples").resolve():
        raise HTTPException(status_code=404, detail="Đường dẫn mẫu không hợp lệ.")
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh mẫu.")
    return FileResponse(image_path)


@app.post("/api/documents")
async def create_document(
    file: UploadFile = File(...), engine: str = Form("vietocr")
) -> dict:
    if engine not in {"paddle", "vietocr"}:
        raise HTTPException(status_code=422, detail="Engine phải là paddle hoặc vietocr.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Tệp tải lên đang trống.")
    if len(content) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413, detail=f"Ảnh vượt quá giới hạn {limit_mb:g} MB."
        )

    try:
        from io import BytesIO

        with Image.open(BytesIO(content)) as image:
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=415, detail="Tệp tải lên không phải ảnh hợp lệ.") from error

    extension = ALLOWED_FORMATS.get(image_format or "")
    if extension is None:
        raise HTTPException(status_code=415, detail="Định dạng ảnh chưa được hỗ trợ.")

    document_id = str(uuid.uuid4())
    image_name = f"{document_id}{extension}"
    image_path = database.UPLOAD_DIR / image_name
    image_path.write_bytes(content)
    try:
        text, detections = await run_in_threadpool(
            recognize, str(image_path), engine
        )
        database.save_document(document_id, image_name, text, engine, detections)
    except OCRUnavailableError as error:
        image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        image_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500, detail=f"Nhận dạng OCR thất bại: {error}"
        ) from error

    result = database.get_document(document_id)
    if result is None:
        raise HTTPException(status_code=500, detail="Không đọc lại được kết quả OCR.")
    return result


@app.get("/api/documents/{document_id}/image")
async def document_image(document_id: str) -> FileResponse:
    document = database.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
    image_path = database.UPLOAD_DIR / document["image_name"]
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh gốc.")
    return FileResponse(image_path)


@app.post("/api/documents/{document_id}/regions")
async def recognize_missing_region(
    document_id: str, selection: RegionSelection
) -> dict[str, Any]:
    document = database.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
    image_path = database.UPLOAD_DIR / document["image_name"]
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh gốc.")

    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            right = selection.x + selection.width
            bottom = selection.y + selection.height
            if right > image.width or bottom > image.height:
                raise HTTPException(
                    status_code=422,
                    detail="Vùng chọn nằm ngoài kích thước ảnh gốc.",
                )
            crop = image.crop((selection.x, selection.y, right, bottom))

        temporary_path = database.DATA_DIR / f".region-{uuid.uuid4()}.png"
        try:
            crop.save(temporary_path, format="PNG")
            text, detections = await run_in_threadpool(
                recognize, str(temporary_path), document["engine"]
            )
        finally:
            temporary_path.unlink(missing_ok=True)
    except OCRUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Không nhận dạng được vùng chọn: {error}"
        ) from error

    if not text.strip() or not detections:
        raise HTTPException(
            status_code=422,
            detail="Không nhận ra chữ trong vùng chọn. Hãy chọn sát phần có chữ hơn.",
        )

    for detection in detections:
        detection["box"] = [
            [point[0] + selection.x, point[1] + selection.y]
            for point in detection["box"]
        ]
    updated_document = database.append_document_region(
        document_id, text.strip(), detections, selection.edited_text
    )
    if updated_document is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
    updated_document["added_text"] = text.strip()
    return updated_document


@app.post("/api/documents/{document_id}/detections/adjust")
async def adjust_detection(
    document_id: str, adjustment: DetectionAdjustment
) -> dict[str, Any]:
    document = database.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
    if adjustment.detection_index >= len(document["detections"]):
        raise HTTPException(status_code=422, detail="Dòng OCR được chọn không hợp lệ.")
    image_path = database.UPLOAD_DIR / document["image_name"]
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh gốc.")

    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            right = adjustment.x + adjustment.width
            bottom = adjustment.y + adjustment.height
            if right > image.width or bottom > image.height:
                raise HTTPException(
                    status_code=422,
                    detail="Vùng điều chỉnh nằm ngoài kích thước ảnh gốc.",
                )
            crop = image.crop((adjustment.x, adjustment.y, right, bottom))

        temporary_path = database.DATA_DIR / f".adjust-{uuid.uuid4()}.png"
        try:
            crop.save(temporary_path, format="PNG")
            text, _ = await run_in_threadpool(
                recognize, str(temporary_path), document["engine"]
            )
        finally:
            temporary_path.unlink(missing_ok=True)
    except OCRUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Không nhận dạng được vùng điều chỉnh: {error}"
        ) from error

    corrected_text = text.strip()
    if not corrected_text:
        raise HTTPException(
            status_code=422,
            detail="Không nhận ra chữ trong vùng mới. Hãy chọn lại sát dòng chữ hơn.",
        )
    if "\n" in corrected_text or "\r" in corrected_text:
        raise HTTPException(
            status_code=422,
            detail="Vùng mới chứa nhiều dòng. Hãy chọn lại quanh đúng một dòng OCR.",
        )

    detection = {
        "box": [
            [float(adjustment.x), float(adjustment.y)],
            [float(right), float(adjustment.y)],
            [float(right), float(bottom)],
            [float(adjustment.x), float(bottom)],
        ],
        "text": corrected_text,
        "confidence": None,
        "manually_adjusted": True,
    }
    updated_document = database.update_document_detection(
        document_id,
        adjustment.detection_index,
        detection,
        corrected_text,
        adjustment.edited_text,
        adjustment.edited_line_index,
    )
    if updated_document is None:
        raise HTTPException(status_code=422, detail="Không thể cập nhật dòng OCR.")

    detector_dir = database.DATA_DIR / "detection_samples"
    detector_dir.mkdir(parents=True, exist_ok=True)
    sample_id = str(uuid.uuid4())
    relative_path = Path("detection_samples") / f"{sample_id}.png"
    sample_path = detector_dir / f"{sample_id}.png"
    crop.save(sample_path, format="PNG")
    fingerprint = hashlib.sha256(
        f"{crop.width}x{crop.height}:RGB:".encode("ascii") + crop.tobytes()
    ).hexdigest()
    points = [
        [0.0, 0.0],
        [float(crop.width), 0.0],
        [float(crop.width), float(crop.height)],
        [0.0, float(crop.height)],
    ]
    stored_id, is_new = database.save_detection_sample(
        document_id,
        adjustment.detection_index,
        str(relative_path),
        fingerprint,
        points,
        corrected_text,
    )
    if not is_new:
        sample_path.unlink(missing_ok=True)
    updated_document["adjusted_text"] = corrected_text
    updated_document["detection_sample_id"] = stored_id
    updated_document["detection_sample_created"] = is_new
    return updated_document


@app.put("/api/documents/{document_id}")
async def update_document(
    document_id: str, correction: Correction
) -> dict[str, str | int]:
    document = database.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
    image_path = database.UPLOAD_DIR / document["image_name"]
    samples = await run_in_threadpool(
        extract_corrected_samples,
        image_path,
        document["detections"],
        document["original_text"],
        correction.edited_text,
        database.DATA_DIR,
        document["engine"],
    )
    if not database.update_document(document_id, correction.edited_text):
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
    sample_counts = database.replace_training_samples(document_id, samples)
    if sample_counts is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
    saved_samples, duplicate_samples, conflicted_samples = sample_counts
    return {
        "status": "reviewed",
        "message": (
            f"Đã lưu bản chỉnh sửa; thêm {saved_samples} mẫu dòng mới, "
            f"gộp {duplicate_samples} crop trùng"
            + (
                f"; {conflicted_samples} mẫu có nhãn mâu thuẫn cần được kiểm tra."
                if conflicted_samples
                else "."
            )
        ),
        "training_samples_created": saved_samples,
        "training_samples_skipped_duplicate": duplicate_samples,
        "training_samples_with_conflicting_labels": conflicted_samples,
    }
