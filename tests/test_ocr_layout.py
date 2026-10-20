import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app import database, main
from app.ocr import (
    DEWARP_MAX_DISPLACEMENT_RATIO,
    DEWARP_MIN_DISPLACEMENT_RATIO,
    PageTransform,
    _align_page,
    _correct_page_orientation,
    _detect_text_regions,
    _dewarp_page,
    _estimate_skew_angle,
    _find_page_quad,
    _detect_layout_regions,
    _layout_blocks_from_detections,
    _layout_regions_from_detections,
    _map_lines_to_original,
    _merge_layout_headings,
    _order_layout_regions,
    _order_text_regions,
    _order_contact_label_values,
    _should_use_ppocr_for_layout_blocks,
    _preprocess_tesseract_region,
    _get_text_detector,
    _get_tesseract,
    _recognize_tesseract_text_regions,
    _recognize_tesseract_layout_ppocr_regions,
    _recognize_tesseract_table_block,
    _recognize_tesseract_table_grid,
    _recognize_tesseract_numeric_region,
    _numeric_table_region_ids,
    _expand_table_block_to_top_rule,
    _expand_table_block_to_grid,
    _find_table_grid,
    _find_table_grid_line_segments,
    _merge_table_grid_cells,
    _filter_small_overlapping_text_regions,
    _order_table_text_regions,
    OCRUnavailableError,
    detect_layout_boxes,
    recognize,
)
from app.pdf_text import merge_pdf_text


class OCRLayoutTests(unittest.TestCase):
    def test_tesseract_uses_conda_vietnamese_data_when_prefix_is_unset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tessdata = Path(temp_dir) / "share" / "tessdata"
            tessdata.mkdir(parents=True)
            (tessdata / "vie.traineddata").touch()

            with (
                patch("app.ocr.sys.prefix", temp_dir),
                patch.dict(os.environ, {"TESSDATA_PREFIX": ""}),
                patch("pytesseract.get_tesseract_version"),
                patch("pytesseract.get_languages", return_value=["eng", "vie"]),
            ):
                _get_tesseract.__wrapped__()

                self.assertEqual(os.environ["TESSDATA_PREFIX"], str(tessdata))

    def test_upload_format_accepts_mpo_photos_with_jpg_extension(self) -> None:
        self.assertEqual(main.ALLOWED_FORMATS["MPO"], ".jpg")

    def test_auto_router_uses_doclayout_blocks_when_regions_are_available(self) -> None:
        skills_page_regions = [
            (99, 80, 163, 54),
            (100, 152, 300, 37),
            (906, 153, 195, 37),
            (913, 200, 423, 36),
            (118, 201, 463, 35),
            (913, 233, 463, 37),
            (118, 234, 646, 69),
            (913, 268, 391, 36),
            (118, 301, 631, 69),
            (913, 302, 285, 34),
            (117, 367, 606, 36),
            (117, 403, 537, 34),
            (103, 457, 455, 52),
            (116, 528, 984, 32),
            (116, 562, 924, 32),
            (111, 616, 408, 52),
            (107, 688, 1466, 131),
        ]
        wall_of_text_regions = [
            (100, top, 1450, 30)
            for top in (100, 135, 170, 205, 240, 275)
        ]

        self.assertTrue(
            _should_use_ppocr_for_layout_blocks(
                [(*region, "text") for region in skills_page_regions],
                (1600, 900),
            )
        )
        self.assertFalse(
            _should_use_ppocr_for_layout_blocks(
                [(*region, "text") for region in wall_of_text_regions],
                (1600, 900),
            )
        )

    def test_auto_router_uses_ppocr_for_two_column_layout(self) -> None:
        blocks = [
            (20, 20, 120, 25, "text"),
            (20, 60, 180, 25, "list item"),
            (260, 20, 120, 25, "text"),
            (260, 60, 180, 25, "list item"),
        ]

        self.assertTrue(_should_use_ppocr_for_layout_blocks(blocks, (500, 300)))

    def test_auto_router_does_not_use_ppocr_for_many_single_column_lines(self) -> None:
        blocks = [
            (100, top, 900, 24, "text")
            for top in range(50, 50 + 42 * 32, 32)
        ]

        self.assertFalse(
            _should_use_ppocr_for_layout_blocks(blocks, (1200, 1500))
        )

    def test_layout_detection_deduplicates_near_identical_boxes(self) -> None:
        detections = [
            (20, 20, 240, 100, 0.9, "text"),
            (21, 21, 241, 101, 0.8, "list item"),
            (20, 110, 240, 130, 0.7, "table"),
        ]

        blocks = _layout_blocks_from_detections(detections, (300, 200))

        self.assertEqual(len(blocks), 2)
        self.assertEqual({block[4] for block in blocks}, {"text", "table"})

    def test_doclayout_regions_read_skill_column_before_next_section(self) -> None:
        regions = [
            (99, 80, 163, 54),
            (100, 152, 300, 37),
            (906, 153, 195, 37),
            (913, 200, 423, 36),
            (118, 201, 463, 35),
            (913, 233, 463, 37),
            (118, 234, 646, 69),
            (913, 268, 391, 36),
            (118, 301, 631, 69),
            (913, 302, 285, 34),
            (117, 367, 606, 36),
            (117, 403, 537, 34),
            (103, 457, 455, 52),
            (116, 528, 984, 32),
            (116, 562, 924, 32),
            (111, 616, 408, 52),
            (107, 688, 1466, 131),
        ]

        ordered = _order_layout_regions(regions)

        self.assertEqual(
            ordered,
            [
                regions[index]
                for index in [
                    0, 1, 4, 6, 8, 10, 11, 2, 3, 5, 7, 9, 12, 13, 14, 15, 16
                ]
            ],
        )

    def test_pdf_text_replaces_matching_ocr_without_changing_ocr_order(self) -> None:
        class PageRect:
            width = 200
            height = 100

        class Page:
            rect = PageRect()

            def get_text(self, _kind, sort=False):
                self.sort = sort
                return [
                    (10, 10, 55, 20, "Xây", 0, 0, 0),
                    (60, 10, 90, 20, "dựng", 0, 0, 1),
                    (95, 10, 140, 20, "marketing.", 0, 0, 2),
                    (110, 60, 150, 70, "Kinh", 1, 0, 0),
                    (155, 60, 190, 70, "nghiệm", 1, 0, 1),
                ]

        left_box = [[10, 10], [140, 10], [140, 20], [10, 20]]
        right_box = [[110, 60], [190, 60], [190, 70], [110, 70]]
        detections = [
            {"text": "Kinh nghiêm", "box": right_box},
            {"text": "Xây dựng meorketing.", "box": left_box},
        ]

        text, merged = merge_pdf_text(Page(), detections)

        self.assertEqual(text, "Kinh nghiệm\nXây dựng marketing.")
        self.assertEqual([item["text_source"] for item in merged], ["pdf", "pdf"])
        self.assertEqual(merged[0]["box"], right_box)

    def test_pdf_text_replaces_content_match_even_when_positions_do_not_match(self) -> None:
        class Page:
            def get_text(self, _kind, sort=False):
                return [(10, 10, 70, 20, "PDF text", 0, 0, 0)]

        detections = [
            {
                "text": "PDF tex",
                "box": [[100, 60], [180, 60], [180, 70], [100, 70]],
            }
        ]

        text, merged = merge_pdf_text(Page(), detections)

        self.assertEqual(text, "PDF text")
        self.assertEqual(merged[0]["text_source"], "pdf")

    def test_pdf_text_replaces_ocr_with_best_matching_segment_only(self) -> None:
        class Page:
            def get_text(self, _kind, sort=False):
                return [
                    (10, 10, 60, 20, "Coventry", 0, 0, 0),
                    (65, 10, 120, 20, "University", 0, 0, 1),
                    (125, 10, 165, 20, "Coventry,", 0, 0, 2),
                    (170, 10, 200, 20, "United", 0, 0, 3),
                    (205, 10, 250, 20, "Kingdom", 0, 0, 4),
                ]

        detections = [{"text": "Coventry University"}]

        text, merged = merge_pdf_text(Page(), detections)

        self.assertEqual(text, "Coventry University")
        self.assertEqual(merged[0]["text"], "Coventry University")
        self.assertEqual(merged[0]["text_source"], "pdf")

    def test_pdf_text_keeps_unrelated_ocr_when_content_does_not_match(self) -> None:
        class Page:
            def get_text(self, _kind, sort=False):
                return [(10, 10, 70, 20, "PDF text", 0, 0, 0)]

        detections = [{"text": "Unrelated OCR"}]

        text, merged = merge_pdf_text(Page(), detections)

        self.assertEqual(text, "Unrelated OCR")
        self.assertNotIn("text_source", merged[0])

    def test_pdf_page_pipeline_uses_matching_text_without_reordering_ocr(self) -> None:
        import pymupdf

        pdf = pymupdf.open()
        page = pdf.new_page(width=300, height=200)
        page.insert_text((20, 40), "Alpha project heading")
        page.insert_text((20, 100), "Beta experience summary")
        words_by_line = {}
        for word in page.get_text("words", sort=False):
            words_by_line.setdefault((word[5], word[6]), []).append(word)
        detections = []
        for words in words_by_line.values():
            words.sort(key=lambda word: word[7])
            source_text = " ".join(word[4] for word in words)
            ocr_text = source_text.replace("project", "projct").replace(
                "experience", "experence"
            )
            detections.append(
                {
                    "text": ocr_text,
                    "box": [[0, 0], [1, 0], [1, 1], [0, 1]],
                }
            )
        detections.reverse()
        pdf_bytes = pdf.tobytes()
        pdf.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(database, "DATA_DIR", Path(temp_dir)),
                patch.object(main, "recognize", return_value=("ignored", detections)),
            ):
                results = main._recognize_pdf_pages(pdf_bytes, "tesseract")

        self.assertEqual(
            results[0][1],
            "Beta experience summary\nAlpha project heading",
        )

    def test_layout_detector_keeps_text_regions_and_clips_to_page(self) -> None:
        regions = _layout_regions_from_detections(
            [
                (-10, 20, 100, 80, 0.9, "Text"),
                (200, 30, 300, 90, 0.8, "Caption"),
                (10, 10, 50, 50, 0.19, "title"),
                (0, 0, 180, 180, 0.95, "figure"),
            ],
            (250, 150),
        )

        self.assertEqual(regions, [(0, 17, 100, 66), (200, 27, 50, 66)])

    def test_layout_regions_use_reduced_vertical_padding_to_avoid_overlap(self) -> None:
        for second_top, expected in (
            (25, [(0, 8, 100, 14), (0, 23, 100, 14)]),
            (22, [(0, 9, 100, 12), (0, 21, 100, 12)]),
        ):
            with self.subTest(second_top=second_top):
                regions = _layout_regions_from_detections(
                    [
                        (0, 10, 100, 20, 0.9, "Text"),
                        (0, second_top, 100, second_top + 10, 0.9, "Text"),
                    ],
                    (120, 80),
                )

                self.assertEqual(regions, expected)

    def test_layout_title_joins_following_aligned_text_block(self) -> None:
        blocks = _layout_blocks_from_detections(
            [
                (11, 8, 549, 30, 0.92, "Section-header"),
                (105, 11, 436, 30, 0.22, "Section-header"),
                (11, 54, 674, 117, 0.91, "Text"),
            ],
            (707, 395),
        )

        merged = _merge_layout_headings(blocks)

        self.assertEqual(merged, [(11, 5, 663, 115, "text")])

    def test_centered_layout_title_joins_following_full_width_text_block(self) -> None:
        blocks = [
            (109, 13, 385, 22, "section header"),
            (14, 62, 768, 71, "text"),
        ]

        self.assertEqual(
            _merge_layout_headings(blocks),
            [(14, 13, 768, 120, "text")],
        )

    def test_layout_title_does_not_join_unaligned_body_or_other_heading(self) -> None:
        blocks = [
            (10, 10, 300, 24, "section header"),
            (400, 48, 250, 70, "text"),
            (10, 180, 300, 24, "section header"),
        ]

        self.assertEqual(_merge_layout_headings(blocks), blocks)

    def test_table_box_replaces_nested_text_boxes(self) -> None:
        regions = _layout_regions_from_detections(
            [
                (30, 20, 250, 180, 0.8, "table"),
                (40, 30, 100, 25, 0.9, "title"),
                (50, 80, 180, 30, 0.7, "Text"),
            ],
            (400, 300),
        )

        self.assertEqual(regions, [(30, 17, 220, 166)])

    def test_table_block_expands_to_outer_grid_lines(self) -> None:
        image = Image.new("RGB", (200, 100), "white")
        grid = ([0, 10, 30, 80, 160, 168], [0, 10, 20, 40, 78, 84])

        with patch("app.ocr._find_table_grid", return_value=grid):
            expanded = _expand_table_block_to_grid(
                image, (30, 20, 140, 60, "table")
            )

        self.assertEqual(expanded, (26, 18, 150, 68, "table"))

    def test_table_grid_line_segments_keep_merged_cell_gaps(self) -> None:
        image = Image.new("RGB", (100, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.line((10, 10, 90, 10), fill="black")
        draw.line((10, 40, 90, 40), fill="black")
        draw.line((10, 70, 90, 70), fill="black")
        draw.line((10, 90, 90, 90), fill="black")
        draw.line((10, 10, 10, 90), fill="black")
        draw.line((50, 40, 50, 90), fill="black")
        draw.line((90, 10, 90, 90), fill="black")

        lines = _find_table_grid_line_segments(image)
        vertical_segments = sorted(
            (round(line["box"][0][0]), round(line["box"][0][1]), round(line["box"][1][1]))
            for line in lines
            if round(line["box"][0][0]) == round(line["box"][1][0])
        )

        self.assertIn((50, 41, 91), vertical_segments)
        self.assertNotIn((50, 11, 41), vertical_segments)

    def test_table_block_expands_to_page_edges_when_grid_reaches_edges(self) -> None:
        image = Image.new("RGB", (1123, 574), "white")
        grid = ([0, 20, 1000, 1123], [0, 100, 500, 574])

        with patch("app.ocr._find_table_grid", return_value=grid):
            expanded = _expand_table_block_to_grid(
                image, (18, 0, 1052, 574, "table")
            )

        self.assertEqual(expanded, (0, 0, 1123, 574, "table"))

    def test_layout_boxes_map_back_to_original_image_coordinates(self) -> None:
        image = Image.new("RGB", (200, 100), "white")
        aligned_image = Image.new("RGB", (200, 100), "white")
        transform = np.array(
            [[1, 0, 10], [0, 1, 5], [0, 0, 1]], dtype=np.float32
        )
        with (
            patch(
                "app.ocr._align_page",
                return_value=(aligned_image, transform),
            ) as align_page,
            patch(
                "app.ocr._detect_layout_blocks",
                return_value=[(30, 20, 40, 30, "table")],
            ),
            patch(
                "app.ocr._find_table_grid",
                return_value=([0, 20, 40], [0, 15, 30]),
            ),
            patch(
                "app.ocr._find_table_grid_line_segments",
                return_value=[
                    {"box": [[0.0, 0.0], [40.0, 0.0]]},
                    {"box": [[20.0, 0.0], [20.0, 30.0]]},
                ],
            ),
        ):
            boxes = detect_layout_boxes(image)

        self.assertEqual(align_page.call_args.args[0].size, image.size)
        self.assertEqual(
            boxes,
            [
                {
                    "label": "table",
                    "box": [
                        [20.0, 15.0],
                        [60.0, 15.0],
                        [60.0, 45.0],
                        [20.0, 45.0],
                    ],
                    "lines": [
                        [[20.0, 15.0], [60.0, 15.0]],
                        [[40.0, 15.0], [40.0, 45.0]],
                    ],
                }
            ],
        )

    def test_table_regions_are_read_row_by_row_then_left_to_right(self) -> None:
        regions = [
            {"text": "", "box": [[100, 10], [145, 10], [145, 20], [100, 20]]},
            {"text": "", "box": [[10, 10], [50, 10], [50, 20], [10, 20]]},
            {"text": "", "box": [[10, 30], [60, 30], [60, 40], [10, 40]]},
            {"text": "", "box": [[100, 30], [150, 30], [150, 40], [100, 40]]},
        ]

        ordered = _order_table_text_regions(regions)

        self.assertEqual(ordered, [regions[1], regions[0], regions[2], regions[3]])

    def test_table_order_groups_cells_with_different_text_heights_by_baseline(
        self,
    ) -> None:
        regions = [
            {"text": "", "box": [[100, 11], [145, 11], [145, 19], [100, 19]]},
            {"text": "", "box": [[10, 5], [50, 5], [50, 25], [10, 25]]},
            {"text": "", "box": [[100, 38], [150, 38], [150, 52], [100, 52]]},
            {"text": "", "box": [[10, 35], [60, 35], [60, 55], [10, 55]]},
        ]

        ordered = _order_table_text_regions(regions)

        self.assertEqual(
            ordered, [regions[1], regions[0], regions[3], regions[2]]
        )

    def test_complex_table_regions_are_read_row_by_row(self) -> None:
        regions = [
            {"text": "left second", "box": [[10, 47], [45, 47], [45, 59], [10, 59]]},
            {"text": "right first", "box": [[70, 31], [95, 31], [95, 43], [70, 43]]},
            {"text": "left first", "box": [[10, 30], [45, 30], [45, 42], [10, 42]]},
        ]

        ordered = _order_table_text_regions(regions)

        self.assertEqual(
            [region["text"] for region in ordered],
            ["left first", "right first", "left second"],
        )

    def test_table_grid_finds_row_and_column_rules(self) -> None:
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        cv2.line(image, (0, 30), (119, 30), (0, 0, 0), 1)
        cv2.line(image, (0, 60), (119, 60), (0, 0, 0), 1)
        cv2.line(image, (40, 0), (40, 79), (0, 0, 0), 1)
        cv2.line(image, (80, 0), (80, 79), (0, 0, 0), 1)

        grid = _find_table_grid(Image.fromarray(image))

        self.assertIsNotNone(grid)
        self.assertEqual(grid[0], [0, 40, 80, 120])
        self.assertEqual(grid[1], [0, 30, 60, 80])

    def test_table_grid_merges_cells_when_shared_rule_is_absent(self) -> None:
        image = Image.new("RGB", (100, 80), "white")
        draw = ImageDraw.Draw(image)
        draw.line((10, 10, 90, 10), fill="black")
        draw.line((10, 40, 90, 40), fill="black")
        draw.line((10, 70, 90, 70), fill="black")
        draw.line((10, 10, 10, 70), fill="black")
        draw.line((50, 40, 50, 70), fill="black")
        draw.line((90, 10, 90, 70), fill="black")
        draw.line((50, 14, 50, 36), fill="black")

        cells = _merge_table_grid_cells(
            image, ([10, 50, 90], [10, 40, 70])
        )

        self.assertEqual(cells, [(10, 10, 90, 40), (10, 40, 50, 70), (50, 40, 90, 70)])

    def test_table_grid_merges_framed_cells_across_all_rows_and_columns(self) -> None:
        image = Image.new("RGB", (101, 81), "white")
        draw = ImageDraw.Draw(image)
        columns = [0, 25, 50, 75, 100]
        rows = [0, 20, 40, 60, 80]

        for y in rows:
            for left, right in zip(columns, columns[1:]):
                if y == 40 and left == 75:
                    continue
                draw.line((left, y, right, y), fill="black")
        for x in columns:
            for top, bottom in zip(rows, rows[1:]):
                if (top == 0 and x in (25, 50, 75)) or (
                    top == 20 and x == 50
                ) or (top == 60 and x == 25):
                    continue
                draw.line((x, top, x, bottom), fill="black")

        cells = _merge_table_grid_cells(
            image, (columns, rows)
        )

        self.assertEqual(
            cells,
            [
                (0, 0, 100, 20),
                (0, 20, 25, 40),
                (25, 20, 75, 40),
                (75, 20, 100, 60),
                (0, 40, 25, 60),
                (25, 40, 50, 60),
                (50, 40, 75, 60),
                (0, 60, 50, 80),
                (50, 60, 75, 80),
                (75, 60, 100, 80),
            ],
        )

    def test_table_grid_recognition_uses_merged_cell_crops(self) -> None:
        image = Image.new("RGB", (100, 80), "white")
        draw = ImageDraw.Draw(image)
        draw.line((10, 10, 90, 10), fill="black")
        draw.line((10, 40, 90, 40), fill="black")
        draw.line((10, 70, 90, 70), fill="black")
        draw.line((10, 10, 10, 70), fill="black")
        draw.line((50, 40, 50, 70), fill="black")
        draw.line((90, 10, 90, 70), fill="black")
        cell_result = [
            {"text": "cell", "box": [[0, 0], [7, 0], [7, 7], [0, 7]]}
        ]

        with patch(
            "app.ocr._recognize_tesseract_page",
            return_value=cell_result,
        ) as recognize_page:
            lines = _recognize_tesseract_table_grid(
                object(),
                image,
                ([10, 50, 90], [10, 40, 70]),
            )

        self.assertEqual(len(lines), 3)
        self.assertEqual(
            [call.args[1].size for call in recognize_page.call_args_list],
            [(74, 24), (34, 24), (34, 24)],
        )

    def test_table_block_expands_up_to_bordered_top_rule(self) -> None:
        image = Image.new("RGB", (120, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.line((5, 20, 114, 20), fill="black", width=1)
        draw.line((5, 40, 114, 40), fill="black", width=1)
        draw.line((5, 90, 114, 90), fill="black", width=1)
        draw.line((5, 20, 5, 90), fill="black", width=1)
        draw.line((114, 20, 114, 90), fill="black", width=1)

        expanded = _expand_table_block_to_top_rule(
            image, (5, 40, 110, 50, "table")
        )

        self.assertEqual(expanded, (5, 18, 110, 72, "table"))

    def test_table_block_ignores_unrelated_top_horizontal_rule(self) -> None:
        image = Image.new("RGB", (120, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.line((5, 20, 114, 20), fill="black", width=1)

        self.assertEqual(
            _expand_table_block_to_top_rule(
                image, (5, 40, 110, 50, "table")
            ),
            (5, 40, 110, 50, "table"),
        )

    def test_table_grid_recognition_reads_cells_in_row_major_order(self) -> None:
        image = Image.new("RGB", (120, 80), "white")
        cell_lines = [
            [{"text": f"cell {index}", "box": [[0, 0], [7, 0], [7, 7], [0, 7]]}]
            for index in range(9)
        ]
        with patch(
            "app.ocr._recognize_tesseract_page",
            side_effect=cell_lines,
        ) as recognize_page:
            lines = _recognize_tesseract_table_grid(
                object(),
                image,
                ([0, 40, 80, 120], [0, 30, 60, 80]),
            )

        self.assertEqual(
            [line["text"] for line in lines],
            [f"cell {index}" for index in range(9)],
        )
        self.assertEqual(
            [line["box"][0] for line in lines[:4]],
            [[3, 3], [43, 3], [83, 3], [3, 33]],
        )
        self.assertTrue(
            all(
                call.kwargs["page_segmentation_mode"] == 6
                for call in recognize_page.call_args_list
            )
        )

    def test_table_grid_can_read_direct_cells_in_column_major_order(self) -> None:
        image = Image.new("RGB", (120, 80), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 39, 29), fill="red")
        draw.rectangle((40, 0, 79, 29), fill="green")
        draw.rectangle((0, 30, 39, 59), fill="blue")
        draw.rectangle((40, 30, 79, 59), fill="yellow")
        labels_by_color = {
            (255, 0, 0): "top-left",
            (0, 128, 0): "top-right",
            (0, 0, 255): "bottom-left",
            (255, 255, 0): "bottom-right",
        }

        def recognize_cell(_tesseract, cell_image, **_kwargs):
            label = labels_by_color[cell_image.getpixel((0, 0))]
            return [
                {
                    "text": label,
                    "box": [[0, 0], [7, 0], [7, 7], [0, 7]],
                }
            ]

        with patch(
            "app.ocr._merge_table_grid_cells",
            return_value=[
                (0, 0, 40, 30),
                (40, 0, 80, 30),
                (0, 30, 40, 60),
                (40, 30, 80, 60),
            ],
        ), patch(
            "app.ocr._recognize_tesseract_page",
            side_effect=recognize_cell,
        ) as recognize_page:
            lines = _recognize_tesseract_table_grid(
                object(),
                image,
                ([0, 40, 80], [0, 30, 60]),
                use_ppocr_regions=False,
                column_major=True,
            )

        self.assertEqual(
            [line["text"] for line in lines],
            ["top-left", "bottom-left", "top-right", "bottom-right"],
        )
        self.assertEqual(
            [
                call.args[1].getpixel((0, 0))
                for call in recognize_page.call_args_list
            ],
            [
                (255, 0, 0),
                (0, 0, 255),
                (0, 128, 0),
                (255, 255, 0),
            ],
        )

    def test_table_grid_uses_line_detections_per_cell_before_cell_fallback(
        self,
    ) -> None:
        image = Image.new("RGB", (100, 60), "white")
        regions = [
            {"text": "", "box": [[60, 10], [80, 10], [80, 18], [60, 18]]},
            {"text": "", "box": [[10, 10], [30, 10], [30, 18], [10, 18]]},
            {"text": "", "box": [[60, 40], [80, 40], [80, 48], [60, 48]]},
        ]
        line_result = {
            "text": "detected",
            "box": [[3, 3], [10, 3], [10, 10], [3, 10]],
        }
        fallback_result = {
            "text": "fallback",
            "box": [[0, 0], [7, 0], [7, 7], [0, 7]],
        }
        with (
            patch(
                "app.ocr._recognize_tesseract_text_regions",
                return_value=[line_result],
            ) as recognize_regions,
            patch(
                "app.ocr._recognize_tesseract_page",
                return_value=[fallback_result],
            ) as recognize_cell,
        ):
            lines = _recognize_tesseract_table_grid(
                object(),
                image,
                ([0, 50, 100], [0, 30, 60]),
                regions,
            )

        self.assertEqual(
            [line["text"] for line in lines],
            ["detected", "detected", "fallback", "detected"],
        )
        self.assertEqual(recognize_regions.call_count, 3)
        recognize_cell.assert_called_once()

    def test_table_grid_keeps_regions_overlapping_the_three_pixel_cell_margin(
        self,
    ) -> None:
        image = Image.new("RGB", (100, 30), "white")
        draw = ImageDraw.Draw(image)
        draw.line((0, 0, 100, 0), fill="black")
        draw.line((0, 29, 100, 29), fill="black")
        draw.line((0, 0, 0, 29), fill="black")
        draw.line((50, 0, 50, 29), fill="black")
        draw.line((99, 0, 99, 29), fill="black")
        regions = [
            {
                "text": "",
                "box": [[47, 10], [55, 10], [55, 18], [47, 18]],
                "confidence": 0.9,
            }
        ]
        line_result = {
            "text": "edge",
            "box": [[0, 0], [7, 0], [7, 7], [0, 7]],
        }

        with (
            patch(
                "app.ocr._recognize_tesseract_text_regions",
                return_value=[line_result],
            ) as recognize_regions,
            patch("app.ocr._recognize_tesseract_page", return_value=[]),
        ):
            lines = _recognize_tesseract_table_grid(
                object(),
                image,
                ([0, 50, 100], [0, 30]),
                regions,
            )

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["text"], "edge")
        region_image, local_regions = recognize_regions.call_args.args[1:3]
        self.assertEqual(region_image.size, (50, 24))
        self.assertEqual(min(point[0] for point in local_regions[0]["box"]), 0)

    def test_table_grid_splits_oversized_pp_regions_into_text_lines(self) -> None:
        image = Image.new("RGB", (100, 60), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 5, 90, 5), fill="black")
        draw.rectangle((10, 54, 90, 54), fill="black")
        draw.rectangle((10, 5, 10, 54), fill="black")
        draw.rectangle((90, 5, 90, 54), fill="black")
        draw.text((30, 15), "HELLO", font=ImageFont.load_default(), fill="black")
        draw.text((30, 34), "WORLD", font=ImageFont.load_default(), fill="black")
        region = {
            "box": [[25, 12], [65, 12], [65, 48], [25, 48]],
            "confidence": 0.9,
        }
        recognized_line = {
            "text": "line",
            "box": [[0, 0], [7, 0], [7, 7], [0, 7]],
        }

        with (
            patch(
                "app.ocr._recognize_tesseract_text_regions",
                return_value=[recognized_line],
            ) as recognize_regions,
            patch("app.ocr._recognize_tesseract_page", return_value=[]),
        ):
            lines = _recognize_tesseract_table_grid(
                object(),
                image,
                ([10, 90], [5, 55]),
                [region],
            )

        self.assertEqual(len(lines), 2)
        self.assertEqual(recognize_regions.call_count, 2)
        detected_bands = [
            (
                min(point[1] for point in call.args[2][0]["box"]),
                max(point[1] for point in call.args[2][0]["box"]),
            )
            for call in recognize_regions.call_args_list
        ]
        self.assertLess(detected_bands[0][1], detected_bands[1][0])

    def test_table_grid_splits_lines_with_two_pixel_vertical_gap(self) -> None:
        image = Image.new("RGB", (100, 40), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 0, 90, 0), fill="black")
        draw.rectangle((10, 39, 90, 39), fill="black")
        draw.rectangle((10, 0, 10, 39), fill="black")
        draw.rectangle((90, 0, 90, 39), fill="black")
        draw.rectangle((32, 10, 40, 13), fill="black")
        draw.rectangle((32, 16, 40, 19), fill="black")
        region = {
            "box": [[25, 8], [55, 8], [55, 22], [25, 22]],
            "confidence": 0.9,
        }

        with (
            patch(
                "app.ocr._recognize_tesseract_text_regions",
                return_value=[
                    {
                        "text": "line",
                        "box": [[0, 0], [7, 0], [7, 7], [0, 7]],
                    }
                ],
            ) as recognize_regions,
            patch("app.ocr._recognize_tesseract_page", return_value=[]),
        ):
            lines = _recognize_tesseract_table_grid(
                object(),
                image,
                ([10, 90], [0, 40]),
                [region],
            )

        self.assertEqual(len(lines), 2)
        self.assertEqual(recognize_regions.call_count, 2)

    def test_table_grid_discards_only_regions_tiny_in_both_dimensions(self) -> None:
        image = Image.new("RGB", (100, 40), "white")
        regions = [
            {
                "box": [[20, 12], [23, 12], [23, 15], [20, 15]],
                "confidence": 0.9,
            },
            {
                "box": [[24, 12], [30, 12], [30, 18], [24, 18]],
                "confidence": 0.9,
            },
            {
                "box": [[30, 10], [32, 10], [32, 20], [30, 20]],
                "confidence": 0.9,
            },
            {
                "box": [[38, 10], [45, 10], [45, 17], [38, 17]],
                "confidence": 0.9,
            },
            {
                "box": [[60, 12], [72, 12], [72, 14], [60, 14]],
                "confidence": 0.9,
            },
        ]

        with (
            patch(
                "app.ocr._recognize_tesseract_text_regions",
                return_value=[
                    {
                        "text": "kept",
                        "box": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    }
                ],
            ) as recognize_regions,
            patch(
                "app.ocr._recognize_tesseract_page",
                return_value=[
                    {
                        "text": "fallback",
                        "box": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    }
                ],
            ) as recognize_cell,
        ):
            _recognize_tesseract_table_grid(
                object(),
                image,
                ([0, 50, 100], [0, 40]),
                regions,
            )

        self.assertEqual(recognize_regions.call_count, 3)
        recognize_cell.assert_not_called()

    def test_table_grid_filters_tiny_tesseract_fallback_result(self) -> None:
        image = Image.new("RGB", (100, 30), "white")
        tiny_line = {
            "text": "noise",
            "box": [[0, 0], [3, 0], [3, 3], [0, 3]],
        }
        narrow_line = {
            "text": "1",
            "box": [[0, 0], [3, 0], [3, 10], [0, 10]],
        }

        with patch(
            "app.ocr._recognize_tesseract_page",
            side_effect=[[tiny_line], [narrow_line]],
        ):
            lines = _recognize_tesseract_table_grid(
                object(),
                image,
                ([0, 50, 100], [0, 30]),
            )

        self.assertEqual([line["text"] for line in lines], ["1"])
        self.assertEqual(lines[0]["box"][0], [53, 3])

    def test_numeric_table_detection_uses_repeated_numeric_text_columns(self) -> None:
        numeric_records = [
            {
                "region": {
                    "box": [[98, y], [112, y], [112, y + 12], [98, y + 12]]
                },
                "lines": [{"text": text}],
            }
            for y, text in ((10, "70"), (30, "T0"), (50, "65"), (70, "52"))
        ]
        text_records = [
            {
                "region": {
                    "box": [[198, y], [212, y], [212, y + 12], [198, y + 12]]
                },
                "lines": [{"text": text}],
            }
            for y, text in ((10, "ns"), (30, "yes"), (50, "ns"), (70, "no"))
        ]
        noisy_numeric_records = [
            {
                "region": {
                    "box": [[148, y], [162, y], [162, y + 12], [148, y + 12]]
                },
                "lines": [{"text": text}],
            }
            for y, text in enumerate(
                ("11", "15", "9", "11", "12", "15", "12", "§", "11", "8",
                 "neutral", "14", "10", "T", "HLÃ", "%")
            )
        ]

        self.assertEqual(
            _numeric_table_region_ids(
                numeric_records + text_records + noisy_numeric_records
            ),
            {
                id(record["region"])
                for record in numeric_records + noisy_numeric_records
            },
        )

    def test_numeric_region_uses_digit_allowlist_and_maps_box_back(self) -> None:
        tesseract = Mock()
        tesseract.Output.DICT = "dict"
        tesseract.image_to_data.return_value = {
            "text": ["22"],
            "conf": ["90"],
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
            "left": [4],
            "top": [5],
            "width": [12],
            "height": [8],
        }

        lines = _recognize_tesseract_numeric_region(
            tesseract,
            Image.new("RGB", (100, 60), "white"),
            {"box": [[40, 20], [58, 20], [58, 34], [40, 34]]},
        )

        self.assertEqual(lines[0]["text"], "22")
        self.assertEqual(lines[0]["box"][0], [38.0, 18.5])
        self.assertEqual(tesseract.image_to_data.call_args.kwargs["lang"], "eng")
        self.assertIn(
            "tessedit_char_whitelist=0123456789",
            tesseract.image_to_data.call_args.kwargs["config"],
        )

    def test_table_block_uses_detected_text_lines_and_maps_them_to_page(self) -> None:
        image = Image.new("RGB", (300, 200), "white")
        table_regions = [
            {"text": "", "box": [[70, 20], [110, 20], [110, 32], [70, 32]]},
            {"text": "", "box": [[20, 20], [55, 20], [55, 32], [20, 32]]},
        ]
        recognized = [
            {"text": "left cell", "box": [[20, 20], [55, 20], [55, 32], [20, 32]]},
            {"text": "right cell", "box": [[70, 20], [110, 20], [110, 32], [70, 32]]},
        ]
        with (
            patch("app.ocr._dewarp_page", return_value=(image.crop((17, 17, 133, 83)), None)),
            patch("app.ocr._detect_text_regions", return_value=table_regions),
            patch(
                "app.ocr._recognize_tesseract_text_regions",
                return_value=recognized,
            ) as recognize_regions,
        ):
            lines = _recognize_tesseract_table_block(
                object(), image, (20, 20, 110, 60, "table")
            )

        self.assertEqual(
            [line["text"] for line in lines],
            ["left cell", "right cell"],
        )
        self.assertEqual(lines[0]["box"][0], [37, 37])
        self.assertEqual(lines[1]["box"][0], [87, 37])
        self.assertEqual(
            [region["box"][0] for region in recognize_regions.call_args.args[2]],
            [[20, 20], [70, 20]],
        )
        self.assertFalse(recognize_regions.call_args.kwargs["order_contact_values"])

    def test_dense_table_reads_each_cell_without_pp_region_crops(self) -> None:
        image = Image.new("RGB", (300, 200), "white")
        regions = [
            {"box": [[10, 10], [30, 10], [30, 18], [10, 18]], "confidence": 0.9},
            {"box": [[60, 40], [80, 40], [80, 48], [60, 48]], "confidence": 0.8},
        ]
        recognized = [
            {"text": "first cell line", "box": [[3, 3], [23, 3], [23, 11], [3, 11]]},
            {"text": "second row line", "box": [[3, 3], [23, 3], [23, 11], [3, 11]]},
        ]
        dense_grid = (
            list(range(0, 301, 30)),
            list(range(0, 201, 25)),
        )
        with (
            patch("app.ocr._dewarp_page", side_effect=lambda crop: (crop, None)),
            patch("app.ocr._find_table_grid", return_value=dense_grid),
            patch(
                "app.ocr._detect_text_regions",
                side_effect=AssertionError("dense grid should use direct cell OCR"),
            ) as detect_text_regions,
            patch(
                "app.ocr._recognize_tesseract_table_grid",
                return_value=recognized,
            ) as recognize_grid,
        ):
            lines = _recognize_tesseract_table_block(
                object(), image, (20, 20, 250, 160, "table")
            )

        self.assertEqual(lines, recognized)
        detect_text_regions.assert_not_called()
        recognize_grid.assert_called_once()
        self.assertEqual(recognize_grid.call_args.args[2], dense_grid)
        self.assertFalse(recognize_grid.call_args.kwargs["use_ppocr_regions"])

    def test_very_sparse_table_reads_each_cell_without_pp_region_crops(self) -> None:
        image = Image.new("RGB", (300, 200), "white")
        sparse_grid = ([0, 150, 300], [0, 100, 200])
        recognized = [{"text": "cell text", "box": [[3, 3], [30, 3], [30, 12], [3, 12]]}]
        with (
            patch("app.ocr._dewarp_page", side_effect=lambda crop: (crop, None)),
            patch("app.ocr._find_table_grid", return_value=sparse_grid),
            patch(
                "app.ocr._detect_text_regions",
                side_effect=AssertionError("sparse grid should use direct cell OCR"),
            ) as detect_text_regions,
            patch(
                "app.ocr._recognize_tesseract_table_grid",
                return_value=recognized,
            ) as recognize_grid,
        ):
            lines = _recognize_tesseract_table_block(
                object(), image, (20, 20, 250, 160, "table")
            )

        self.assertEqual(lines, recognized)
        detect_text_regions.assert_not_called()
        recognize_grid.assert_called_once()
        self.assertEqual(recognize_grid.call_args.args[2], sparse_grid)
        self.assertFalse(recognize_grid.call_args.kwargs["use_ppocr_regions"])
        self.assertTrue(recognize_grid.call_args.kwargs["column_major"])

    def test_small_pp_regions_overlapped_by_thirty_percent_are_removed(self) -> None:
        regions = [
            {"box": [[0, 0], [20, 0], [20, 20], [0, 20]], "id": "large"},
            {"box": [[0, 0], [10, 0], [10, 10], [0, 10]], "id": "covered"},
            {"box": [[34, 0], [54, 0], [54, 10], [34, 10]], "id": "wide"},
            {"box": [[27, 0], [37, 0], [37, 10], [27, 10]], "id": "at_threshold"},
            {"box": [[60, 0], [80, 0], [80, 10], [60, 10]], "id": "large2"},
            {"box": [[53.1, 0], [62.9, 0], [62.9, 10], [53.1, 10]], "id": "under_threshold"},
        ]

        kept = _filter_small_overlapping_text_regions(regions)

        self.assertEqual(
            [region["id"] for region in kept],
            ["large", "wide", "large2", "under_threshold"],
        )

    def test_auto_ocr_routes_detected_table_blocks_to_table_recognizer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (160, 100), "white")
            image.save(image_path)
            table_block = [(10, 10, 100, 60, "table")]
            expected_lines = [
                {"text": "cell text", "box": [[12, 12], [30, 12], [30, 20], [12, 20]]}
            ]
            with (
                patch("app.ocr._align_page", return_value=(image, np.eye(3))),
                patch("app.ocr._get_tesseract", return_value=object()),
                patch("app.ocr._detect_layout_blocks", return_value=table_block),
                patch("app.ocr._order_layout_blocks", return_value=table_block),
                patch(
                    "app.ocr._recognize_tesseract_layout_with_tables",
                    return_value=expected_lines,
                ) as recognize_tables,
                patch(
                    "app.ocr._should_use_ppocr_for_layout_blocks"
                ) as route_layout,
                patch("app.ocr._map_lines_to_original"),
            ):
                text, lines = recognize(str(image_path), "doclayout_tesseract")

        self.assertEqual(text, "cell text")
        self.assertEqual(lines, expected_lines)
        recognize_tables.assert_called_once()
        self.assertEqual(recognize_tables.call_args.args[2], table_block)
        route_layout.assert_not_called()

    def test_separate_model_regions_remain_separate_and_reading_ordered(self) -> None:
        regions = _layout_regions_from_detections(
            [
                (500, 100, 780, 600, 0.9, "Text"),
                (40, 100, 450, 600, 0.9, "Text"),
            ],
            (900, 1000),
        )

        self.assertEqual(regions, [(40, 97, 410, 506), (500, 97, 280, 506)])

    def test_inference_maps_yolo_box_tensors_to_layout_regions(self) -> None:
        class Tensor:
            def __init__(self, values):
                self.values = values

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        class Boxes:
            xyxy = Tensor([[10.0, 20.0, 210.0, 120.0]])
            conf = Tensor([0.8])
            cls = Tensor([1.0])

            def __len__(self):
                return 1

        class Result:
            boxes = Boxes()
            names = {1: "Text"}

        class Detector:
            def predict(self, image, **kwargs):
                self.image = image
                self.kwargs = kwargs
                return [Result()]

        detector = Detector()
        with patch("app.ocr._get_layout_detector", return_value=(detector, "cpu")):
            regions = _detect_layout_regions(Image.new("RGB", (300, 200), "white"))

        self.assertEqual(regions, [(10, 17, 200, 106)])
        self.assertEqual(detector.image.shape, (200, 300, 3))
        self.assertEqual(detector.kwargs["device"], "cpu")

    def test_preprocessing_enhances_faint_text_and_rejects_blank_noise(self) -> None:
        blank_image = Image.fromarray(np.full((80, 240, 3), 240, dtype=np.uint8))
        blank, _ = _preprocess_tesseract_region(blank_image, 15)
        self.assertIsNone(blank)

        faint_image = np.full((80, 240, 3), 230, dtype=np.uint8)
        faint_image[35:45, 20:220] = 205
        enhanced, scale = _preprocess_tesseract_region(
            Image.fromarray(faint_image), 15
        )

        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.mode, "L")
        self.assertEqual(scale, 1.5)

    def test_page_orientation_correction_handles_quarter_turns(self) -> None:
        image_array = np.full((20, 30, 3), 255, dtype=np.uint8)
        image_array[3, 5] = (255, 0, 0)
        image = Image.fromarray(image_array)
        osd_output = (
            "Rotate: 90\nOrientation confidence: 12.0\n"
        )
        with patch("pytesseract.image_to_osd", return_value=osd_output):
            corrected, transform = _correct_page_orientation(image)

        self.assertEqual(corrected.size, (20, 30))
        self.assertEqual(corrected.getpixel((16, 5)), (255, 0, 0))
        point = np.asarray([[[5, 3]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, transform)
        self.assertEqual(mapped[0, 0].tolist(), [16.0, 5.0])

    def test_page_orientation_ignores_low_confidence_osd(self) -> None:
        image = Image.new("RGB", (30, 20), "white")
        with patch(
            "pytesseract.image_to_osd",
            return_value="Rotate: 90\nOrientation confidence: 2.0\n",
        ):
            corrected, transform = _correct_page_orientation(image)

        self.assertIs(corrected, image)
        np.testing.assert_array_equal(transform, np.eye(3, dtype=np.float32))

    def test_page_deskew_accepts_large_arbitrary_rotation(self) -> None:
        angle = 62.0
        radians = np.deg2rad(angle)
        rotation = np.asarray(
            [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
        )
        rotated = (np.asarray([[100, 100], [300, 100]]) - 200) @ rotation.T + 200
        segment = np.round(rotated).astype(np.int32).reshape(1, 1, 4)
        with patch("app.ocr.cv2.HoughLinesP", return_value=segment):
            estimated = _estimate_skew_angle(
                np.full((500, 500, 3), 255, dtype=np.uint8)
            )

        self.assertIsNotNone(estimated)
        self.assertAlmostEqual(abs(estimated), angle, delta=1.0)

    def test_page_dewarp_mapping_restores_original_line_coordinates(self) -> None:
        line = {
            "text": "test",
            "box": [[10.0, 20.0], [30.0, 20.0], [30.0, 30.0], [10.0, 30.0]],
        }
        transform = PageTransform(
            np.eye(3, dtype=np.float32),
            np.linspace(0.0, 10.0, 41, dtype=np.float32),
        )

        _map_lines_to_original([line], transform, (41, 50))

        self.assertEqual(line["box"][0], [10.0, 22.5])
        self.assertEqual(line["box"][1], [30.0, 27.5])

    def test_page_dewarp_corrects_synthetic_curved_text_lines(self) -> None:
        width, height = 1200, 700
        source = Image.new("RGB", (width, height), "white")
        pixels = source.load()
        for line_index, top in enumerate(range(80, 620, 75)):
            for left in range(40, width - 40, 24):
                glyph_width = 10 + (left // 24 + line_index) % 5
                for x in range(left, min(width - 30, left + glyph_width)):
                    for y in range(top, top + 14):
                        pixels[x, y] = (0, 0, 0)

        straight = np.asarray(source)
        yy, xx = np.indices((height, width), dtype=np.float32)
        curve = 20 * np.sin(2 * np.pi * xx / width)
        curved = cv2.remap(
            straight,
            xx,
            yy + curve,
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        corrected, offsets = _dewarp_page(Image.fromarray(curved))

        self.assertIsNotNone(offsets)
        self.assertGreater(
            float(np.ptp(offsets)),
            height * DEWARP_MIN_DISPLACEMENT_RATIO,
        )
        self.assertLessEqual(
            float(np.max(np.abs(offsets))),
            height * DEWARP_MAX_DISPLACEMENT_RATIO + 1,
        )
        self.assertEqual(corrected.size, (width, height))

    def test_page_dewarp_leaves_straight_or_blank_page_unchanged(self) -> None:
        image = Image.new("RGB", (500, 400), "white")
        corrected, offsets = _dewarp_page(image)

        self.assertIs(corrected, image)
        self.assertIsNone(offsets)

    def test_page_alignment_dewarps_full_page_before_layout_detection(self) -> None:
        image = Image.new("RGB", (400, 300), "white")
        offsets = np.linspace(0.0, 5.0, image.width, dtype=np.float32)
        with (
            patch("app.ocr._correct_page_orientation", return_value=(image, np.eye(3))),
            patch("app.ocr._find_page_quad", return_value=None),
            patch("app.ocr._estimate_skew_angle", return_value=None),
            patch(
                "app.ocr._dewarp_page",
                return_value=(image, offsets),
            ) as dewarp,
        ):
            aligned, transform = _align_page(image)

        self.assertEqual(aligned.size, image.size)
        self.assertEqual(aligned.getpixel((0, 0)), image.getpixel((0, 0)))
        np.testing.assert_array_equal(transform.dewarp_offsets, offsets)
        dewarp.assert_called_once()
        self.assertEqual(dewarp.call_args.args[0].size, image.size)

    def test_page_quad_fallback_finds_bright_page_touching_image_edge(self) -> None:
        image = np.full((400, 300, 3), 60, dtype=np.uint8)
        page = np.array([[280, 10], [20, 20], [0, 399], [290, 399]])
        cv2.fillConvexPoly(image, page, (180, 180, 180))

        quad = _find_page_quad(image)

        self.assertIsNotNone(quad)
        self.assertEqual(quad.shape, (4, 2))
        self.assertGreater(float(quad[:, 0].max()), 280)

    def test_page_coordinates_reverse_global_dewarp_after_frame_dewarp(self) -> None:
        line = {
            "text": "test",
            "box": [[10.0, 20.0], [30.0, 20.0], [30.0, 30.0], [10.0, 30.0]],
        }
        frame_offsets = np.full(41, 2.0, dtype=np.float32)
        page_offsets = np.full(41, 5.0, dtype=np.float32)
        _map_lines_to_original(
            [line],
            PageTransform(np.eye(3, dtype=np.float32), frame_offsets),
            (41, 50),
        )
        _map_lines_to_original(
            [line],
            PageTransform(np.eye(3, dtype=np.float32), page_offsets),
            (41, 50),
        )

        self.assertEqual(line["box"][0], [10.0, 27.0])
        self.assertEqual(line["box"][2], [30.0, 37.0])

    def test_layout_ppocr_dewarps_only_the_detected_region_and_maps_boxes_back(self) -> None:
        image = Image.new("RGB", (100, 100), "white")
        warped_crop = Image.new("RGB", (44, 44), "white")
        detected = [
            {
                "box": [[4, 8], [30, 8], [30, 16], [4, 16]],
                "confidence": 0.9,
            }
        ]
        recognized = [
            {
                "text": "sample",
                "box": [[5, 10], [25, 10], [25, 18], [5, 18]],
            }
        ]
        with (
            patch(
                "app.ocr._dewarp_page",
                return_value=(warped_crop, np.full(44, 5, dtype=np.float32)),
            ) as dewarp,
            patch(
                "app.ocr._detect_text_regions",
                return_value=detected,
            ) as detect,
            patch(
                "app.ocr._recognize_tesseract_text_regions",
                return_value=recognized,
            ) as recognize_regions,
        ):
            lines = _recognize_tesseract_layout_ppocr_regions(
                object(), image, [(20, 20, 40, 40)]
            )

        self.assertEqual(dewarp.call_args.args[0].size, (44, 44))
        detect.assert_called_once_with(warped_crop)
        recognize_regions.assert_called_once()
        self.assertEqual(lines[0]["box"][0], [23.0, 33.0])
        self.assertEqual(lines[0]["box"][2], [43.0, 41.0])

    def test_text_line_crop_adds_vertical_context_for_accent_marks(self) -> None:
        image = Image.new("RGB", (200, 150), "white")
        region = {
            "box": [[20, 50], [100, 50], [100, 80], [20, 80]],
            "confidence": 0.9,
        }
        expected_line = {
            "text": "recognized",
            "box": [[2, 2], [8, 2], [8, 8], [2, 8]],
        }

        with (
            patch(
                "app.ocr._preprocess_tesseract_region",
                side_effect=lambda crop, _height: (crop, 1.0),
            ),
            patch(
                "app.ocr._recognize_tesseract_page",
                side_effect=lambda *_, **__: [
                    {
                        "text": expected_line["text"],
                        "box": [point[:] for point in expected_line["box"]],
                    }
                ],
            ) as recognize_page,
        ):
            lines = _recognize_tesseract_text_regions(
                object(), image, [region]
            )

        self.assertEqual(recognize_page.call_args.args[1].size, (88, 36))
        self.assertEqual(lines[0]["box"][0], [18, 49])

    def test_text_line_vertical_context_stays_clear_of_adjacent_rows(self) -> None:
        image = Image.new("RGB", (200, 150), "white")
        regions = [
            {
                "box": [[20, 50], [100, 50], [100, 80], [20, 80]],
                "confidence": 0.9,
            },
            {
                "box": [[20, 84], [100, 84], [100, 114], [20, 114]],
                "confidence": 0.9,
            },
        ]
        expected_line = {
            "text": "recognized",
            "box": [[2, 2], [8, 2], [8, 8], [2, 8]],
        }

        with (
            patch(
                "app.ocr._preprocess_tesseract_region",
                side_effect=lambda crop, _height: (crop, 1.0),
            ),
            patch(
                "app.ocr._recognize_tesseract_page",
                side_effect=lambda *_, **__: [
                    {
                        "text": expected_line["text"],
                        "box": [point[:] for point in expected_line["box"]],
                    }
                ],
            ) as recognize_page,
        ):
            lines = _recognize_tesseract_text_regions(
                object(), image, regions
            )

        crops = [call.args[1] for call in recognize_page.call_args_list]
        self.assertEqual([crop.size for crop in crops], [(88, 32), (88, 32)])
        self.assertEqual(lines[0]["box"][0], [18, 51])

    def test_ppocrv6_detector_returns_clipped_polygons_and_scores(self) -> None:
        class Detector:
            def predict(self, image, batch_size):
                self.image = image
                self.batch_size = batch_size
                return [
                    SimpleNamespace(
                        json={
                            "res": {
                                "dt_polys": [
                                    [[-2, 10], [40, 10], [40, 30], [-2, 30]],
                                    [[50, 40], [130, 40], [130, 60], [50, 60]],
                                ],
                                "dt_scores": [0.91, 0.82],
                            }
                        }
                    )
                ]

        detector = Detector()
        image = Image.new("RGB", (100, 80), (210, 180, 150))
        with patch("app.ocr._get_text_detector", return_value=detector):
            regions = _detect_text_regions(image)

        self.assertEqual(detector.image.shape, (80, 100, 3))
        np.testing.assert_array_equal(detector.image[0, 0], [150, 180, 210])
        self.assertEqual(detector.batch_size, 1)
        self.assertEqual(
            regions,
            [
                {
                    "box": [[0.0, 10.0], [40.0, 10.0], [40.0, 30.0], [0.0, 30.0]],
                    "confidence": 0.91,
                },
                {
                    "box": [
                        [50.0, 40.0],
                        [99.0, 40.0],
                        [99.0, 60.0],
                        [50.0, 60.0],
                    ],
                    "confidence": 0.82,
                },
            ],
        )

    def test_text_detector_initializes_ppocrv6_on_available_gpu(self) -> None:
        detector = object()
        text_detection = Mock(return_value=detector)
        paddle = SimpleNamespace(
            is_compiled_with_cuda=lambda: True,
            device=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1)),
        )
        modules = {
            "paddle": paddle,
            "paddleocr": SimpleNamespace(TextDetection=text_detection),
        }
        _get_text_detector.cache_clear()
        try:
            with patch.dict(sys.modules, modules):
                result = _get_text_detector()
        finally:
            _get_text_detector.cache_clear()

        self.assertIs(result, detector)
        text_detection.assert_called_once_with(
            model_name="PP-OCRv6_medium_det",
            device="gpu:0",
        )

    def test_ppocrv6_detector_regions_are_grouped_by_layout_then_reading_order(
        self,
    ) -> None:
        text_regions = [
            {"box": [[70, 10], [110, 10], [110, 20], [70, 20]], "confidence": 0.9},
            {"box": [[5, 30], [55, 30], [55, 40], [5, 40]], "confidence": 0.8},
            {"box": [[5, 10], [55, 10], [55, 20], [5, 20]], "confidence": 0.7},
        ]
        layout_regions = [(0, 0, 60, 50), (60, 0, 60, 50)]

        ordered = _order_text_regions(text_regions, layout_regions)

        self.assertEqual(
            [region["confidence"] for region in ordered],
            [0.7, 0.8, 0.9],
        )

    def test_text_regions_are_ordered_left_to_right_within_each_row(self) -> None:
        text_regions = [
            {
                "box": [[60, 9], [100, 9], [100, 19], [60, 19]],
                "confidence": 0.9,
            },
            {
                "box": [[5, 12], [45, 12], [45, 22], [5, 22]],
                "confidence": 0.8,
            },
            {
                "box": [[5, 30], [45, 30], [45, 40], [5, 40]],
                "confidence": 0.7,
            },
        ]

        ordered = _order_text_regions(text_regions, [])

        self.assertEqual(
            [region["confidence"] for region in ordered],
            [0.8, 0.9, 0.7],
        )

    def test_text_regions_are_read_down_each_column_before_next_column(self) -> None:
        text_regions = [
            {"text": "right second", "confidence": 0.9, "box": [[220, 32], [300, 32], [300, 44], [220, 44]]},
            {"text": "left first", "confidence": 0.7, "box": [[10, 10], [90, 10], [90, 22], [10, 22]]},
            {"text": "right first", "confidence": 0.8, "box": [[220, 10], [300, 10], [300, 22], [220, 22]]},
            {"text": "left third", "confidence": 0.6, "box": [[10, 54], [90, 54], [90, 66], [10, 66]]},
            {"text": "right third", "confidence": 1.0, "box": [[220, 54], [300, 54], [300, 66], [220, 66]]},
            {"text": "left second", "confidence": 0.5, "box": [[10, 32], [90, 32], [90, 44], [10, 44]]},
        ]

        expected_text = [
            "left first",
            "left second",
            "left third",
            "right first",
            "right second",
            "right third",
        ]
        for layout_regions in ([], [(0, 0, 320, 75)]):
            with self.subTest(layout_regions=layout_regions):
                ordered = _order_text_regions(text_regions, layout_regions)
                self.assertEqual(
                    [region["text"] for region in ordered],
                    expected_text,
                )

    def test_text_regions_are_column_ordered_when_layout_splits_each_row(self) -> None:
        text_regions = [
            {"text": "right second", "box": [[220, 32], [300, 32], [300, 44], [220, 44]]},
            {"text": "left first", "box": [[10, 10], [90, 10], [90, 22], [10, 22]]},
            {"text": "right first", "box": [[220, 10], [300, 10], [300, 22], [220, 22]]},
            {"text": "left third", "box": [[10, 54], [90, 54], [90, 66], [10, 66]]},
            {"text": "right third", "box": [[220, 54], [300, 54], [300, 66], [220, 66]]},
            {"text": "left second", "box": [[10, 32], [90, 32], [90, 44], [10, 44]]},
        ]
        layout_regions = [
            (10, 10, 80, 12),
            (220, 10, 80, 12),
            (10, 32, 80, 12),
            (220, 32, 80, 12),
            (10, 54, 80, 12),
            (220, 54, 80, 12),
        ]

        ordered = _order_text_regions(text_regions, layout_regions)

        self.assertEqual(
            [region["text"] for region in ordered],
            [
                "left first",
                "left second",
                "left third",
                "right first",
                "right second",
                "right third",
            ],
        )

    def test_two_column_block_precedes_following_full_width_section(self) -> None:
        def region(text, left, top, right, bottom):
            return {
                "text": text,
                "box": [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                ],
            }

        text_regions = [
            region("KỸ NĂNG", 103, 90, 252, 121),
            region("Kỹ năng chuyên môn", 102, 159, 393, 185),
            region("Kỹ năng mềm", 907, 157, 1094, 187),
            region("Sử dụng tốt SEO Google Analytics", 153, 207, 574, 229),
            region("Làm việc nhóm lãnh đạo nhóm", 950, 208, 1328, 233),
            region("Sáng tạo nội dung Quản lý mạng xã hội", 153, 232, 756, 267),
            region("Facebook Zalo OA", 153, 276, 381, 297),
            region("Giải quyết vấn đề Quản lý thời gian", 949, 237, 1372, 263),
            region("Giao tiếp tốt Chịu áp lực cao", 949, 271, 1296, 297),
            region("Sử dụng thành thạo Canva Adobe Photoshop", 153, 305, 755, 332),
            region("Lightroom", 153, 342, 269, 366),
            region("Sáng tạo chủ động", 949, 307, 1191, 332),
            region("Quay phim Chỉnh sửa video Thiết kế ấn phẩm", 152, 371, 714, 397),
            region("Tổ chức sự kiện Lập kế hoạch chiến dịch", 152, 405, 645, 431),
            region("CHỨNG CHỈ VÀ NGÔN NGỮ", 104, 464, 549, 497),
            region("Ngôn Ngữ Tiếng Anh Tiếng Trung", 118, 532, 1090, 556),
            region("MỤC TIÊU NGHỀ NGHIỆP", 115, 623, 512, 661),
            region("Tôi định hướng phát triển trong lĩnh vực Digital Marketing", 108, 691, 1564, 714),
        ]
        expected = [
            "KỸ NĂNG",
            "Kỹ năng chuyên môn",
            "Sử dụng tốt SEO Google Analytics",
            "Sáng tạo nội dung Quản lý mạng xã hội",
            "Facebook Zalo OA",
            "Sử dụng thành thạo Canva Adobe Photoshop",
            "Lightroom",
            "Quay phim Chỉnh sửa video Thiết kế ấn phẩm",
            "Tổ chức sự kiện Lập kế hoạch chiến dịch",
            "Kỹ năng mềm",
            "Làm việc nhóm lãnh đạo nhóm",
            "Giải quyết vấn đề Quản lý thời gian",
            "Giao tiếp tốt Chịu áp lực cao",
            "Sáng tạo chủ động",
            "CHỨNG CHỈ VÀ NGÔN NGỮ",
            "Ngôn Ngữ Tiếng Anh Tiếng Trung",
            "MỤC TIÊU NGHỀ NGHIỆP",
            "Tôi định hướng phát triển trong lĩnh vực Digital Marketing",
        ]

        ordered = _order_text_regions(
            text_regions, [(100, 80, 1470, 750)]
        )

        self.assertEqual(
            [region["text"] for region in ordered],
            expected,
        )

    def test_contact_labels_stay_with_values_before_next_section(self) -> None:
        def region(text, left, top, right, bottom):
            return {
                "text": text,
                "box": [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                ],
            }

        text_regions = [
            region("LÊ NGUYỄN HẢI LONG", 441, 88, 1033, 145),
            region("Address", 438, 191, 568, 213),
            region("161 Hoa Bằng, Phường Yên Hòa, Hà Nội", 626, 186, 1190, 214),
            region("Phone", 439, 235, 538, 257),
            region("0862-166-776", 634, 230, 840, 253),
            region("Email", 439, 278, 523, 301),
            region("longle@example.com", 635, 272, 1029, 300),
            region("Linkedin", 439, 322, 569, 345),
            region("https://example.com/profile", 635, 313, 1324, 341),
            region("GIỚI THIỆU BẢN THÂN", 100, 408, 460, 446),
            region("Giới thiệu nội dung", 100, 468, 1554, 526),
            region("HỌC VẤN", 101, 668, 254, 708),
            region("Đại học RMIT Việt Nam", 98, 747, 418, 775),
        ]

        ordered = _order_text_regions(text_regions, [])

        self.assertEqual(
            [region["text"] for region in ordered],
            [
                "LÊ NGUYỄN HẢI LONG",
                "Address",
                "161 Hoa Bằng, Phường Yên Hòa, Hà Nội",
                "Phone",
                "0862-166-776",
                "Email",
                "longle@example.com",
                "Linkedin",
                "https://example.com/profile",
                "GIỚI THIỆU BẢN THÂN",
                "Giới thiệu nội dung",
                "HỌC VẤN",
                "Đại học RMIT Việt Nam",
            ],
        )

    def test_contact_labels_are_paired_with_values_after_column_reading(self) -> None:
        def line(text, left, top, right, bottom):
            return {
                "text": text,
                "box": [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom],
                ],
            }

        lines = [
            line("LÊ NGUYỄN HẢI LONG", 441, 88, 1033, 145),
            line("Address:", 438, 191, 568, 213),
            line("Phone:", 439, 235, 538, 257),
            line("Email:", 439, 278, 523, 301),
            line("Linkedin:", 439, 322, 569, 345),
            line("161 Hoa Bằng, Phường Yên Hòa, Hà Nội", 626, 186, 1190, 214),
            line("0862-166-776", 634, 230, 840, 253),
            line("longle.lnh29.04@gmail.com", 635, 272, 1029, 300),
            line(
                "https://www.linkedin.com/in/long-le-5120a3285",
                635,
                313,
                1324,
                341,
            ),
            line("GIỚI THIỆU BẢN THÂN", 100, 408, 460, 446),
        ]

        ordered = _order_contact_label_values(lines)

        self.assertEqual(
            [line["text"] for line in ordered],
            [
                "LÊ NGUYỄN HẢI LONG",
                "Address:",
                "161 Hoa Bằng, Phường Yên Hòa, Hà Nội",
                "Phone:",
                "0862-166-776",
                "Email:",
                "longle.lnh29.04@gmail.com",
                "Linkedin:",
                "https://www.linkedin.com/in/long-le-5120a3285",
                "GIỚI THIỆU BẢN THÂN",
            ],
        )

    def test_ppocrv6_hybrid_uses_tesseract_for_each_detected_text_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            source_pixels = np.full((80, 120, 3), 255, dtype=np.uint8)
            source_pixels[10:20, 5:115] = 0
            source_pixels[30:40, 5:55] = 0
            source_image = Image.fromarray(source_pixels)
            source_image.save(image_path)
            expected_lines = [
                {"text": "left first", "box": [[5, 8], [15, 8], [15, 15], [5, 15]]},
                {"text": "left second", "box": [[5, 8], [15, 8], [15, 15], [5, 15]]},
                {"text": "right", "box": [[5, 8], [15, 8], [15, 15], [5, 15]]},
            ]
            with (
                patch(
                    "app.ocr._align_page",
                    return_value=(source_image, np.eye(3)),
                ),
                patch(
                    "app.ocr._get_layout_detector",
                    return_value=(object(), "cpu"),
                ),
                patch(
                    "app.ocr._detect_layout_regions",
                    return_value=[(0, 0, 60, 50), (60, 0, 60, 50)],
                ),
                patch("app.ocr._get_tesseract", return_value=object()),
                patch(
                    "app.ocr._recognize_tesseract_layout_ppocr_regions",
                    return_value=expected_lines,
                ) as recognize_regions,
                patch("app.ocr._map_lines_to_original"),
            ):
                text, lines = recognize(str(image_path), "ppocrv6")

        self.assertEqual(
            text,
            "left first\nleft second\nright",
        )
        self.assertEqual(len(lines), len(expected_lines))
        self.assertEqual(
            [line["text"] for line in lines],
            [line["text"] for line in expected_lines],
        )
        recognize_regions.assert_called_once()
        self.assertIs(recognize_regions.call_args.args[1], source_image)
        self.assertEqual(
            recognize_regions.call_args.args[2],
            [(0, 0, 60, 50), (60, 0, 60, 50)],
        )

    def test_ppocrv6_falls_back_to_full_page_tesseract_when_detection_is_empty(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (120, 80), "white").save(image_path)
            with (
                patch(
                    "app.ocr._align_page",
                    return_value=(Image.new("RGB", (120, 80), "white"), np.eye(3)),
                ),
                patch("app.ocr._get_layout_detector", return_value=(object(), "cpu")),
                patch("app.ocr._get_tesseract", return_value=object()),
                patch("app.ocr._detect_text_regions", return_value=[]),
                patch("app.ocr._detect_layout_regions", return_value=[]),
                patch(
                    "app.ocr._recognize_tesseract_page",
                    return_value=[{"text": "fallback"}],
                ) as recognize_page,
                patch("app.ocr._map_lines_to_original"),
            ):
                text, lines = recognize(str(image_path), "ppocrv6")

        self.assertEqual(text, "fallback")
        self.assertEqual(len(lines), 1)
        self.assertEqual(len(recognize_page.call_args.args), 2)
        self.assertEqual(recognize_page.call_args.args[1].size, (120, 80))

    def test_doclayout_ppocr_mode_runs_ppocr_with_doclayout_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (160, 100), "white")
            image.save(image_path)
            layout_blocks = [
                (10, 10, 60, 70, "text"),
                (90, 10, 60, 70, "text"),
            ]
            layout_regions = [(x, y, width, height) for x, y, width, height, _ in layout_blocks]
            expected_lines = [
                {
                    "text": "CV line",
                    "box": [[12, 15], [58, 15], [58, 22], [12, 22]],
                }
            ]
            with (
                patch("app.ocr._align_page", return_value=(image, np.eye(3))),
                patch(
                    "app.ocr._detect_layout_blocks",
                    return_value=layout_blocks,
                ) as detect_layout,
                patch(
                    "app.ocr._order_layout_blocks",
                    return_value=layout_blocks,
                ),
                patch("app.ocr._get_tesseract", return_value=object()),
                patch(
                    "app.ocr._order_layout_regions",
                    return_value=layout_regions,
                ),
                patch(
                    "app.ocr._recognize_tesseract_layout_ppocr_regions",
                    return_value=expected_lines,
                ) as recognize_regions,
                patch("app.ocr._map_lines_to_original"),
            ):
                text, lines = recognize(
                    str(image_path), "doclayout_ppocr_tesseract"
                )

        self.assertEqual(text, "CV line")
        self.assertEqual(lines, expected_lines)
        detect_layout.assert_called_once_with(image)
        recognize_regions.assert_called_once()
        self.assertIs(recognize_regions.call_args.args[1], image)
        self.assertEqual(recognize_regions.call_args.args[2], layout_regions)

    def test_preprocessing_preserves_normal_contrast_and_keeps_faint_text(self) -> None:
        normal_pixels = np.full((40, 80, 3), 255, dtype=np.uint8)
        normal_pixels[18:23, 10:70] = 0
        normal_image = Image.fromarray(normal_pixels)
        processed, scale = _preprocess_tesseract_region(normal_image, 20)
        self.assertIs(processed, normal_image)
        self.assertEqual(scale, 1.0)

        faint_pixels = np.full((40, 80, 3), 230, dtype=np.uint8)
        faint_pixels[18:23, 10:70] = 218
        faint, faint_scale = _preprocess_tesseract_region(
            Image.fromarray(faint_pixels), 20
        )
        self.assertIsNotNone(faint)
        self.assertEqual(faint.mode, "L")
        self.assertEqual(faint_scale, 1.0)

    def test_empty_layout_result_uses_full_page_without_handcrafted_segmentation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (120, 80), "white").save(image_path)
            expected_lines = [{"text": "sample", "box": [[1, 1], [5, 1], [5, 5], [1, 5]]}]
            with (
                patch("app.ocr._get_tesseract", return_value=object()),
                patch("app.ocr._detect_layout_regions", return_value=[]),
                patch("app.ocr._recognize_tesseract_page", return_value=expected_lines) as recognize_page,
                patch("app.ocr._map_lines_to_original"),
            ):
                text, lines = recognize(str(image_path), "tesseract")

        self.assertEqual(text, "sample")
        self.assertEqual(lines, expected_lines)
        self.assertEqual(recognize_page.call_args.args[1].size, (120, 80))

    def test_tesseract_ocr_runs_once_for_each_model_region(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            Image.new("RGB", (120, 80), "white").save(image_path)
            expected_lines = [
                {"text": "left", "box": [[1, 1], [5, 1], [5, 5], [1, 5]]},
                {"text": "right", "box": [[1, 1], [5, 1], [5, 5], [1, 5]]},
            ]
            with (
                patch("app.ocr._get_tesseract", return_value=object()),
                patch(
                    "app.ocr._detect_layout_regions",
                    return_value=[(10, 10, 30, 20), (60, 10, 30, 20)],
                ),
                patch(
                    "app.ocr._preprocess_tesseract_region",
                    side_effect=lambda image, _: (image, 1.0),
                ),
                patch(
                    "app.ocr._recognize_tesseract_page",
                    side_effect=[[expected_lines[0]], [expected_lines[1]]],
                ) as recognize_page,
                patch("app.ocr._map_lines_to_original"),
            ):
                text, lines = recognize(str(image_path), "tesseract")

        self.assertEqual(text, "left\nright")
        self.assertEqual(lines, expected_lines)
        self.assertEqual(recognize_page.call_count, 2)
        self.assertEqual(
            [call.args[1].size for call in recognize_page.call_args_list],
            [(46, 36), (46, 36)],
        )

    def test_doclayout_block_mode_uses_tesseract_directly_on_ordered_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (160, 100), "white")
            regions = [
                (100, 10, 30, 20),
                (10, 10, 30, 20),
                (100, 40, 30, 20),
                (10, 40, 30, 20),
            ]
            region_colors = {
                (100, 10): (255, 0, 0),
                (10, 10): (0, 255, 0),
                (100, 40): (0, 0, 255),
                (10, 40): (255, 255, 0),
            }
            for left, top, width, height in regions:
                image.paste(
                    region_colors[(left, top)],
                    (left, top, left + width, top + height),
                )
            image.paste((0, 0, 0), (12, 6, 16, 9))
            image.save(image_path)
            recognized_lines = [
                {"text": text, "box": [[0, 0], [10, 0], [10, 10], [0, 10]]}
                for text in ["left first", "left second", "right first", "right second"]
            ]
            with (
                patch(
                    "app.ocr._align_page",
                    return_value=(image, np.eye(3)),
                ),
                patch("app.ocr._get_tesseract", return_value=object()),
                patch(
                    "app.ocr._detect_layout_blocks",
                    return_value=[(*region, "list item") for region in regions],
                ),
                patch("app.ocr._should_use_ppocr_for_layout_blocks", return_value=False),
                patch(
                    "app.ocr._preprocess_tesseract_region",
                    side_effect=lambda crop, _height: (crop, 1.0),
                ),
                patch(
                    "app.ocr._recognize_tesseract_page",
                    side_effect=[[line] for line in recognized_lines],
                ) as recognize_page,
                patch("app.ocr._map_lines_to_original"),
            ):
                text, lines = recognize(str(image_path), "doclayout_tesseract")

        self.assertEqual(
            text,
            "left first\nleft second\nright first\nright second",
        )
        self.assertEqual(
            [line["text"] for line in lines],
            [line["text"] for line in recognized_lines],
        )
        self.assertEqual(
            [call.args[1].getpixel((10, 10)) for call in recognize_page.call_args_list],
            [
                region_colors[(10, 10)],
                region_colors[(10, 40)],
                region_colors[(100, 10)],
                region_colors[(100, 40)],
            ],
        )
        self.assertEqual(
            [call.args[1].size for call in recognize_page.call_args_list],
            [(46, 36)] * 4,
        )
        self.assertEqual(
            [
                call.args[2]
                for call in recognize_page.call_args_list
            ],
            [11, 6, 11, 6],
        )

    def test_auto_router_uses_ppocr_lines_inside_doclayout_for_complex_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.png"
            image = Image.new("RGB", (160, 100), "white")
            image.save(image_path)
            blocks = [
                (10, 10, 50, 30, "text"),
                (100, 10, 50, 30, "text"),
            ]
            expected_lines = [
                {"text": "recognized line", "box": [[12, 15], [58, 15], [58, 22], [12, 22]]}
            ]
            with (
                patch("app.ocr._align_page", return_value=(image, np.eye(3))),
                patch("app.ocr._get_tesseract", return_value=object()),
                patch("app.ocr._detect_layout_blocks", return_value=blocks),
                patch("app.ocr._order_layout_blocks", return_value=blocks),
                patch("app.ocr._should_use_ppocr_for_layout_blocks", return_value=True),
                patch(
                    "app.ocr._recognize_tesseract_layout_ppocr_regions",
                    return_value=expected_lines,
                ) as recognize_regions,
                patch("app.ocr._recognize_tesseract_layout_blocks") as direct_blocks,
                patch("app.ocr._map_lines_to_original"),
            ):
                text, lines = recognize(str(image_path), "doclayout_tesseract")

        self.assertEqual(text, "recognized line")
        self.assertEqual(lines, expected_lines)
        recognize_regions.assert_called_once()
        self.assertEqual(
            recognize_regions.call_args.args[2],
            [(10, 10, 50, 30), (100, 10, 50, 30)],
        )
        direct_blocks.assert_not_called()

    def test_pdf_upload_directory_is_recreated_before_saving_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            upload_dir = data_dir / "uploads"
            with (
                patch.object(database, "DATA_DIR", data_dir),
                patch.object(database, "UPLOAD_DIR", upload_dir),
                patch.object(database, "DATABASE_PATH", data_dir / "ocr.sqlite3"),
            ):
                database.initialize_database()
                upload_dir.rmdir()

                with patch.object(
                    main,
                    "run_in_threadpool",
                    new=AsyncMock(return_value=[(b"png-data", "page text", [])]),
                ):
                    result = asyncio.run(
                        main._create_pdf_documents(b"pdf-data", "test.pdf", "tesseract")
                    )

                self.assertTrue((upload_dir / result["image_name"]).is_file())
                self.assertEqual(result["original_text"], "page text")

    def test_document_layout_endpoint_returns_layout_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir)
            Image.new("RGB", (40, 30), "white").save(upload_dir / "doc.png")
            expected_boxes = [
                {
                    "label": "table",
                    "box": [[1, 2], [20, 2], [20, 15], [1, 15]],
                }
            ]
            with (
                patch.object(database, "UPLOAD_DIR", upload_dir),
                patch.object(
                    database,
                    "get_document",
                    return_value={"image_name": "doc.png"},
                ),
                patch.object(
                    main,
                    "run_in_threadpool",
                    new=AsyncMock(return_value=expected_boxes),
                ),
            ):
                result = asyncio.run(main.document_layout("document-id"))

        self.assertEqual(result, expected_boxes)


if __name__ == "__main__":
    unittest.main()
