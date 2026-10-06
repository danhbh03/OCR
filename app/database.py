import json
import hashlib
import difflib
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("OCR_DATA_DIR", BASE_DIR / "data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
DATABASE_PATH = DATA_DIR / "ocr.sqlite3"


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "training_samples").mkdir(parents=True, exist_ok=True)
    duplicate_paths: list[str] = []
    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                image_name TEXT NOT NULL,
                original_text TEXT NOT NULL,
                edited_text TEXT NOT NULL,
                engine TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'needs_review',
                detections_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS training_samples (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                label TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                fingerprint TEXT,
                engine TEXT NOT NULL DEFAULT 'paddle',
                label_conflict INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS training_sample_sources (
                sample_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(sample_id, document_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_samples (
                id TEXT PRIMARY KEY,
                image_path TEXT NOT NULL,
                fingerprint TEXT NOT NULL UNIQUE,
                points_json TEXT NOT NULL,
                text TEXT NOT NULL,
                engine TEXT NOT NULL DEFAULT 'paddle-det',
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_sample_sources (
                sample_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                detection_index INTEGER NOT NULL,
                PRIMARY KEY(sample_id, document_id, detection_index)
            )
            """
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(training_samples)")
        }
        if "fingerprint" not in columns:
            connection.execute("ALTER TABLE training_samples ADD COLUMN fingerprint TEXT")
        if "engine" not in columns:
            connection.execute(
                "ALTER TABLE training_samples ADD COLUMN engine TEXT NOT NULL DEFAULT 'paddle'"
            )
        if "label_conflict" not in columns:
            connection.execute(
                "ALTER TABLE training_samples ADD COLUMN label_conflict INTEGER NOT NULL DEFAULT 0"
            )
        detection_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(detection_samples)")
        }
        if "engine" not in detection_columns:
            connection.execute(
                "ALTER TABLE detection_samples ADD COLUMN engine TEXT NOT NULL DEFAULT 'paddle-det'"
            )

        connection.execute(
            """
            UPDATE training_samples SET engine = COALESCE(
                (SELECT engine FROM documents WHERE documents.id = training_samples.document_id),
                'paddle'
            )
            WHERE engine = 'paddle'
              AND EXISTS (
                SELECT 1 FROM documents WHERE documents.id = training_samples.document_id
              )
            """
        )
        source_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(training_sample_sources)")
        }
        if "label" not in source_columns:
            connection.execute(
                "ALTER TABLE training_sample_sources ADD COLUMN label TEXT NOT NULL DEFAULT ''"
            )

        legacy_samples = connection.execute(
            """
            SELECT id, image_path FROM training_samples
            WHERE fingerprint IS NULL ORDER BY created_at, id
            """
        ).fetchall()
        for sample in legacy_samples:
            sample_path = _training_sample_path(sample["image_path"])
            fingerprint = _fingerprint_file(sample_path)
            if fingerprint is None:
                fingerprint = f"missing:{sample['id']}"
            connection.execute(
                "UPDATE training_samples SET fingerprint = ? WHERE id = ?",
                (fingerprint, sample["id"]),
            )

        connection.execute(
            """
            INSERT OR IGNORE INTO training_sample_sources (sample_id, document_id, label)
            SELECT id, document_id, label FROM training_samples
            """
        )
        existing_samples = connection.execute(
            """
            SELECT id, image_path, fingerprint, engine FROM training_samples
            ORDER BY created_at, id
            """
        ).fetchall()
        seen_fingerprints: set[tuple[str, str]] = set()
        duplicates = []
        canonical_by_fingerprint: dict[tuple[str, str], str] = {}
        for sample in existing_samples:
            key = (sample["engine"], sample["fingerprint"])
            if key in seen_fingerprints:
                duplicates.append(sample)
            else:
                seen_fingerprints.add(key)
                canonical_by_fingerprint[key] = sample["id"]
        duplicate_paths = [row["image_path"] for row in duplicates]
        for duplicate in duplicates:
            key = (duplicate["engine"], duplicate["fingerprint"])
            canonical_id = canonical_by_fingerprint[key]
            connection.execute(
                """
                INSERT OR IGNORE INTO training_sample_sources (sample_id, document_id, label)
                SELECT ?, document_id, label FROM training_sample_sources WHERE sample_id = ?
                """,
                (canonical_id, duplicate["id"]),
            )
            connection.execute(
                "DELETE FROM training_sample_sources WHERE sample_id = ?",
                (duplicate["id"],),
            )
            connection.execute(
                "DELETE FROM training_samples WHERE id = ?", (duplicate["id"],)
            )
        connection.execute(
            """
            UPDATE training_samples
            SET label_conflict = (
                SELECT CASE WHEN COUNT(DISTINCT label) > 1 THEN 1 ELSE 0 END
                FROM training_sample_sources
                WHERE training_sample_sources.sample_id = training_samples.id
            )
            """
        )
        connection.execute(
            """
            UPDATE training_samples
            SET label = (
                SELECT label FROM training_sample_sources
                WHERE training_sample_sources.sample_id = training_samples.id
                ORDER BY document_id LIMIT 1
            )
            WHERE label_conflict = 0
              AND EXISTS (
                SELECT 1 FROM training_sample_sources
                WHERE training_sample_sources.sample_id = training_samples.id
            )
            """
        )
        connection.execute(
            """
            DROP INDEX IF EXISTS idx_training_samples_fingerprint
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_training_samples_engine_fingerprint
            ON training_samples(engine, fingerprint)
            """
        )
    _remove_unreferenced_sample_files(duplicate_paths)


def save_document(
    document_id: str,
    image_name: str,
    original_text: str,
    engine: str,
    detections: list[dict[str, Any]],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                id, image_name, original_text, edited_text, engine,
                detections_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                image_name,
                original_text,
                original_text,
                engine,
                json.dumps(detections, ensure_ascii=False),
                now,
                now,
            ),
        )


def get_document(document_id: str) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    return _serialize(row) if row else None


def list_documents() -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        ).fetchall()
    return [_serialize(row) for row in rows]


def update_document(document_id: str, edited_text: str) -> bool:
    with connect() as connection:
        cursor = connection.execute(
            """
            UPDATE documents
            SET edited_text = ?, status = 'reviewed', updated_at = ?
            WHERE id = ?
            """,
            (edited_text, datetime.now(timezone.utc).isoformat(), document_id),
        )
    return cursor.rowcount == 1


def append_document_region(
    document_id: str,
    text: str,
    detections: list[dict[str, Any]],
    edited_text: str | None = None,
) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT original_text, edited_text, detections_json FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            return None

        original_lines = row["original_text"].split("\n")
        current_edited_lines = (
            edited_text if edited_text is not None else row["edited_text"]
        ).split("\n")
        current_detections = json.loads(row["detections_json"])
        entries: list[tuple[dict[str, Any], str, bool, int]] = []
        for index, detection in enumerate(current_detections):
            entries.append(
                (
                    detection,
                    original_lines[index]
                    if index < len(original_lines)
                    else detection.get("text", ""),
                    False,
                    index,
                )
            )
        fallback_lines = text.splitlines()
        for index, detection in enumerate(detections):
            line = str(
                detection.get("text", "")
                or (fallback_lines[index] if index < len(fallback_lines) else "")
            ).strip()
            if not line:
                continue
            new_entry = (detection, line, True, len(entries) + index)
            insertion_position = next(
                (
                    position
                    for position, existing in enumerate(entries)
                    if _comes_before_reading_order(new_entry[0], existing[0])
                ),
                len(entries),
            )
            entries.insert(insertion_position, new_entry)
        insertion_index = next(
            (index for index, entry in enumerate(entries) if entry[2]),
            len(entries),
        )
        ordered_detections = [entry[0] for entry in entries]
        ordered_original_lines = [entry[1] for entry in entries]
        additions_by_boundary: dict[int, list[str]] = {}
        existing_before = 0
        for detection, line, is_added, _ in entries:
            if is_added:
                edited_boundary = _map_line_boundary(
                    original_lines, current_edited_lines, existing_before
                )
                additions_by_boundary.setdefault(edited_boundary, []).append(line)
            else:
                existing_before += 1
        for boundary in sorted(additions_by_boundary, reverse=True):
            current_edited_lines[boundary:boundary] = additions_by_boundary[boundary]
        connection.execute(
            """
            UPDATE documents
            SET original_text = ?, edited_text = ?, detections_json = ?,
                status = 'needs_review', updated_at = ?
            WHERE id = ?
            """,
            (
                "\n".join(ordered_original_lines),
                "\n".join(current_edited_lines),
                json.dumps(ordered_detections, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
                document_id,
            ),
        )
    result = get_document(document_id)
    if result is not None:
        result["inserted_at_line"] = insertion_index
    return result


def _detection_bounds(
    detection: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    points = detection.get("box", [])
    if not points:
        return None
    return (
        min(float(point[0]) for point in points),
        min(float(point[1]) for point in points),
        max(float(point[0]) for point in points),
        max(float(point[1]) for point in points),
    )


def _comes_before_reading_order(
    first: dict[str, Any], second: dict[str, Any]
) -> bool:
    first_bounds = _detection_bounds(first)
    second_bounds = _detection_bounds(second)
    if first_bounds is None:
        return False
    if second_bounds is None:
        return True

    first_left, first_top, first_right, first_bottom = first_bounds
    second_left, second_top, second_right, second_bottom = second_bounds
    first_height = max(1.0, first_bottom - first_top)
    second_height = max(1.0, second_bottom - second_top)
    first_center_y = (first_top + first_bottom) / 2
    second_center_y = (second_top + second_bottom) / 2
    vertical_overlap = max(
        0.0, min(first_bottom, second_bottom) - max(first_top, second_top)
    )
    same_row = (
        vertical_overlap / min(first_height, second_height) >= 0.5
        or abs(first_center_y - second_center_y)
        <= min(first_height, second_height) * 0.4
    )
    if same_row:
        if first_left != second_left:
            return first_left < second_left
        return first_center_y < second_center_y
    return first_center_y < second_center_y


def _map_line_boundary(
    original_lines: list[str], edited_lines: list[str], boundary: int
) -> int:
    matcher = difflib.SequenceMatcher(
        None, original_lines, edited_lines, autojunk=False
    )
    for tag, original_start, original_end, edited_start, edited_end in matcher.get_opcodes():
        if original_start <= boundary <= original_end:
            if tag == "equal":
                return edited_start + boundary - original_start
            original_span = original_end - original_start
            if original_span == 0:
                return edited_end
            offset = boundary - original_start
            return edited_start + round(
                offset * (edited_end - edited_start) / original_span
            )
    return len(edited_lines)


def replace_training_samples(
    document_id: str, samples: list[dict[str, Any]]
) -> tuple[int, int, int] | None:
    now = datetime.now(timezone.utc).isoformat()
    paths_to_clean: list[str] = []
    inserted_count = 0
    duplicate_count = 0
    conflict_count = 0
    with connect() as connection:
        exists = connection.execute(
            "SELECT 1 FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        if exists is None:
            return None
        old_samples = connection.execute(
            """
            SELECT sample.image_path
            FROM training_samples AS sample
            JOIN training_sample_sources AS source ON source.sample_id = sample.id
            WHERE source.document_id = ?
            """,
            (document_id,),
        ).fetchall()
        connection.execute(
            "DELETE FROM training_sample_sources WHERE document_id = ?",
            (document_id,),
        )
        for sample in samples:
            existing = connection.execute(
                """
                SELECT id FROM training_samples
                WHERE engine = ? AND fingerprint = ?
                """,
                (sample["engine"], sample["fingerprint"]),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO training_samples (
                        id, document_id, image_path, label, line_start, line_end,
                        fingerprint, engine, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sample["id"],
                        document_id,
                        sample["image_path"],
                        sample["label"],
                        sample["line_start"],
                        sample["line_end"],
                        sample["fingerprint"],
                        sample["engine"],
                        now,
                    ),
                )
                sample_id = sample["id"]
                inserted_count += 1
            else:
                sample_id = existing["id"]
                duplicate_count += 1
                paths_to_clean.append(sample["image_path"])
            connection.execute(
                """
                INSERT INTO training_sample_sources (sample_id, document_id, label)
                VALUES (?, ?, ?)
                ON CONFLICT(sample_id, document_id)
                DO UPDATE SET label = excluded.label
                """,
                (sample_id, document_id, sample["label"]),
            )
            distinct_labels = connection.execute(
                """
                SELECT DISTINCT label FROM training_sample_sources
                WHERE sample_id = ?
                """,
                (sample_id,),
            ).fetchall()
            labels = {row["label"] for row in distinct_labels}
            if len(labels) == 1:
                connection.execute(
                    """
                    UPDATE training_samples SET label = ?, label_conflict = 0
                    WHERE id = ?
                    """,
                    (next(iter(labels)), sample_id),
                )
            else:
                connection.execute(
                    "UPDATE training_samples SET label_conflict = 1 WHERE id = ?",
                    (sample_id,),
                )
                conflict_count += 1
        paths_to_clean.extend(row["image_path"] for row in old_samples)
        orphaned_samples = connection.execute(
            """
            SELECT id, image_path FROM training_samples
            WHERE NOT EXISTS (
                SELECT 1 FROM training_sample_sources
                WHERE training_sample_sources.sample_id = training_samples.id
            )
            """
        ).fetchall()
        paths_to_clean.extend(row["image_path"] for row in orphaned_samples)
        connection.executemany(
            "DELETE FROM training_samples WHERE id = ?",
            [(row["id"],) for row in orphaned_samples],
        )
    _remove_unreferenced_sample_files(paths_to_clean)
    return inserted_count, duplicate_count, conflict_count


def _training_sample_path(relative_path: str) -> Path:
    sample_dir = (DATA_DIR / "training_samples").resolve()
    path = (DATA_DIR / relative_path).resolve()
    if path.parent != sample_dir:
        raise ValueError(f"Unexpected training sample path: {relative_path}")
    return path


def _fingerprint_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256(
            f"{rgb.width}x{rgb.height}:RGB:".encode("ascii") + rgb.tobytes()
        )
    return digest.hexdigest()


def _remove_unreferenced_sample_files(paths: list[str]) -> None:
    if not paths:
        return
    with connect() as connection:
        referenced = {
            row["image_path"]
            for row in connection.execute("SELECT image_path FROM training_samples")
        }
    for relative_path in set(paths) - referenced:
        _training_sample_path(relative_path).unlink(missing_ok=True)


def list_training_samples() -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, document_id, image_path, label, line_start, line_end, engine,
                   label_conflict, created_at
            FROM training_samples ORDER BY created_at DESC
            """
        ).fetchall()
    return [
        {
            **dict(row),
            "image_url": f"/api/training-samples/{row['id']}/image",
        }
        for row in rows
    ]


def get_training_sample(sample_id: str) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM training_samples WHERE id = ?", (sample_id,)
        ).fetchone()
    return dict(row) if row else None


def save_detection_sample(
    document_id: str,
    detection_index: int,
    image_path: str,
    fingerprint: str,
    points: list[list[float]],
    text: str,
) -> tuple[str, bool]:
    now = datetime.now(timezone.utc).isoformat()
    sample_id = str(uuid.uuid4())
    with connect() as connection:
        existing = connection.execute(
            "SELECT id FROM detection_samples WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO detection_samples (
                    id, image_path, fingerprint, points_json, text, engine, created_at
                ) VALUES (?, ?, ?, ?, ?, 'paddle-det', ?)
                """,
                (
                    sample_id,
                    image_path,
                    fingerprint,
                    json.dumps(points),
                    text,
                    now,
                ),
            )
        else:
            sample_id = existing["id"]
        connection.execute(
            """
            INSERT OR IGNORE INTO detection_sample_sources (
                sample_id, document_id, detection_index
            ) VALUES (?, ?, ?)
            """,
            (sample_id, document_id, detection_index),
        )
    return sample_id, existing is None


def update_document_detection(
    document_id: str,
    detection_index: int,
    detection: dict[str, Any],
    replacement_text: str,
    edited_text: str | None = None,
    edited_line_index: int | None = None,
) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute(
            """
            SELECT original_text, edited_text, detections_json
            FROM documents WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        if row is None:
            return None
        detections = json.loads(row["detections_json"])
        if detection_index >= len(detections):
            return None
        detections[detection_index] = detection
        original_lines = row["original_text"].split("\n")
        if detection_index >= len(original_lines):
            return None
        original_lines[detection_index] = replacement_text
        edited_lines = (
            edited_text if edited_text is not None else row["edited_text"]
        ).split("\n")
        target_line = detection_index if edited_line_index is None else edited_line_index
        if target_line >= len(edited_lines):
            return None
        edited_lines[target_line] = replacement_text
        connection.execute(
            """
            UPDATE documents
            SET original_text = ?, edited_text = ?, detections_json = ?,
                status = 'needs_review', updated_at = ?
            WHERE id = ?
            """,
            (
                "\n".join(original_lines),
                "\n".join(edited_lines),
                json.dumps(detections, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
                document_id,
            ),
        )
    return get_document(document_id)


def list_detection_samples() -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, image_path, points_json, text, engine, created_at
            FROM detection_samples ORDER BY created_at DESC
            """
        ).fetchall()
    return [
        {
            **dict(row),
            "points": json.loads(row["points_json"]),
            "image_url": f"/api/detection-samples/{row['id']}/image",
        }
        for row in rows
    ]


def get_detection_sample(sample_id: str) -> dict[str, Any] | None:
    with connect() as connection:
        row = connection.execute(
            "SELECT * FROM detection_samples WHERE id = ?", (sample_id,)
        ).fetchone()
    return dict(row) if row else None


def _serialize(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["detections"] = json.loads(result.pop("detections_json"))
    result["image_url"] = f"/api/documents/{result['id']}/image"
    return result
