from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field


class CatalogInfoPanelData(BaseModel):
    item_no: str
    name: str = ""
    product_size: str = ""
    thickness: str = ""
    material: str = ""
    hex_approx: str = Field(default="#61615F", pattern=r"^#[0-9A-Fa-f]{6}$")


class CatalogInfoPanelRenderer:
    def render(
        self,
        source: Path,
        target: Path,
        panel: CatalogInfoPanelData,
        *,
        target_usage: str,
    ) -> None:
        with Image.open(source) as opened:
            base = opened.convert("RGBA")
        width, height = base.size
        overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        panel_box = self._panel_box(width, height, target_usage)
        if panel_box is None:
            panel_box = self._compact_panel_box(draw, width, height, panel)
            self._draw_compact_panel(draw, panel_box, panel, width, height)
        else:
            self._draw_infographic_panel(draw, base, panel_box, panel, width, height)
        Image.alpha_composite(base, overlay).convert("RGB").save(target)

    def _panel_box(
        self,
        width: int,
        height: int,
        target_usage: str,
    ) -> tuple[int, int, int, int] | None:
        if target_usage != "detail_infographic":
            return None
        return (
            int(width * 0.672),
            int(height * 0.090),
            int(width * 0.953),
            int(height * 0.261),
        )

    def _draw_infographic_panel(
        self,
        draw: ImageDraw.ImageDraw,
        source: Image.Image,
        box: tuple[int, int, int, int],
        panel: CatalogInfoPanelData,
        width: int,
        height: int,
    ) -> None:
        x1, y1, x2, y2 = box
        panel_height = y2 - y1
        padding = max(17, min(width, height) // 58)
        content_x = x1 + padding
        content_right = x2 - padding
        content_width = content_right - content_x
        self._erase_placeholder_with_surrounding_background(
            draw,
            source,
            box,
            width,
            height,
        )

        title_font = self._font(max(27, panel_height // 6), bold=True)
        body_font = self._font(max(17, panel_height // 10))
        detail_font = self._font(max(17, panel_height // 10), bold=True)
        text_color = (246, 249, 252, 255)
        shadow_color = (7, 10, 14, 170)
        header_gap = max(7, panel_height // 24)
        group_gap = max(11, panel_height // 15)
        detail_gap = max(7, panel_height // 25)

        header_lines: list[tuple[str, ImageFont.ImageFont | ImageFont.FreeTypeFont]] = [
            (f"Color: {panel.item_no}", title_font),
            (
                f"Name: {panel.name or 'Catalog matched automotive film'}",
                body_font,
            ),
        ]
        detail_texts = [
            f"Size: {panel.product_size}" if panel.product_size else "",
            f"Thickness: {panel.thickness}" if panel.thickness else "",
            f"Material: {panel.material}" if panel.material else "",
        ]
        detail_lines = [(line, detail_font) for line in detail_texts if line]

        y = y1 + padding
        lines = [*header_lines, *detail_lines]
        gaps_after = [header_gap, group_gap, detail_gap, detail_gap, 0]
        for index, (line, font) in enumerate(lines):
            fitted = self._fit_text(draw, line, font, content_width)
            draw.text((content_x + 1, y + 1), fitted, font=font, fill=shadow_color)
            draw.text((content_x, y), fitted, font=font, fill=text_color)
            y += self._text_height(draw, fitted, font) + gaps_after[index]

    def _erase_placeholder_with_surrounding_background(
        self,
        draw: ImageDraw.ImageDraw,
        image: Image.Image,
        box: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> None:
        x1, y1, x2, y2 = box
        offset = max(8, min(width, height) // 85)
        erase_x2 = min(width - 1, x2 + 2)
        erase_y2 = min(height - 1, y2 + 2)
        left_x = max(0, x1 - offset)
        right_x = min(width - 1, erase_x2 + offset)
        top_y = max(0, y1 - offset)
        for y in range(y1, erase_y2 + 1):
            left = self._rgb_at(image, left_x, min(height - 1, y))
            right = self._rgb_at(image, right_x, min(height - 1, y))
            top = self._rgb_at(image, x1 + (x2 - x1) // 2, top_y)
            color = (
                (left[0] + right[0] + top[0]) // 3,
                (left[1] + right[1] + top[1]) // 3,
                (left[2] + right[2] + top[2]) // 3,
            )
            draw.line((x1, y, erase_x2, y), fill=color)

    def _rgb_at(self, image: Image.Image, x: int, y: int) -> tuple[int, int, int]:
        pixel = cast(tuple[int, int, int, int], image.getpixel((x, y)))
        return pixel[:3]

    def _draw_metric_tiles(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        metrics: list[tuple[str, str]],
        label_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        value_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    ) -> None:
        x1, y1, x2, y2 = box
        available_metrics = [(label, value) for label, value in metrics if value]
        if not available_metrics:
            return
        gap = 7
        tile_count = len(available_metrics)
        tile_width = max(1, (x2 - x1 - gap * (tile_count - 1)) // tile_count)
        tile_height = y2 - y1
        for index, (label, value) in enumerate(available_metrics):
            tile_x1 = x1 + index * (tile_width + gap)
            tile_x2 = x2 if index == tile_count - 1 else tile_x1 + tile_width
            draw.rounded_rectangle(
                (tile_x1, y1, tile_x2, y2),
                radius=5,
                fill=(41, 50, 60),
                outline=(83, 96, 110),
                width=1,
            )
            label_text = self._fit_text(draw, label, label_font, tile_x2 - tile_x1 - 12)
            value_text = self._fit_text(draw, value, value_font, tile_x2 - tile_x1 - 12)
            label_y = y1 + max(5, tile_height // 7)
            value_y = y2 - self._text_height(draw, value_text, value_font) - max(
                5,
                tile_height // 7,
            )
            draw.text((tile_x1 + 6, label_y), label_text, font=label_font, fill=(150, 162, 176))
            draw.text((tile_x1 + 6, value_y), value_text, font=value_font, fill=(242, 246, 249))

    def _draw_compact_panel(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        panel: CatalogInfoPanelData,
        width: int,
        height: int,
    ) -> None:
        x1, y1, x2, y2 = box
        padding = max(10, min(width, height) // 64)
        draw.rounded_rectangle(
            box,
            radius=max(5, padding // 2),
            fill=(28, 35, 43, 235),
            outline=(92, 102, 114, 210),
            width=1,
        )
        draw.rectangle(
            (x1, y1, x2, y1 + max(6, (y2 - y1) // 12)),
            fill=self._hex_to_rgb(panel.hex_approx),
        )
        font = self._font(max(12, min(width, height) // 56))
        title_font = self._font(max(15, min(width, height) // 44), bold=True)
        y = y1 + padding + 4
        draw.text((x1 + padding, y), panel.item_no, font=title_font, fill=(246, 248, 250))
        y += self._text_height(draw, panel.item_no, title_font) + 5
        for value in [panel.name, panel.product_size, panel.thickness, panel.material]:
            if not value:
                continue
            text = self._fit_text(draw, value, font, x2 - x1 - padding * 2)
            draw.text((x1 + padding, y), text, font=font, fill=(218, 224, 230))
            y += self._text_height(draw, text, font) + 4

    def _compact_panel_box(
        self,
        draw: ImageDraw.ImageDraw,
        width: int,
        height: int,
        panel: CatalogInfoPanelData,
    ) -> tuple[int, int, int, int]:
        font = self._font(max(12, min(width, height) // 56))
        title_font = self._font(max(15, min(width, height) // 44), bold=True)
        lines = [panel.item_no, panel.name, panel.product_size, panel.thickness, panel.material]
        line_width = max((self._text_width(draw, line, font) for line in lines if line), default=0)
        line_width = max(line_width, self._text_width(draw, panel.item_no, title_font))
        padding = max(10, min(width, height) // 64)
        panel_width = min(width - padding * 2, line_width + padding * 2)
        panel_height = max(92, len([line for line in lines if line]) * 22 + padding * 2)
        x1 = padding
        y1 = max(padding, height - panel_height - padding)
        return (
            x1,
            y1,
            min(width - padding, x1 + panel_width),
            min(height - padding, y1 + panel_height),
        )

    def _font(
        self,
        size: int,
        *,
        bold: bool = False,
    ) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
        candidates = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _fit_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
        max_width: int,
    ) -> str:
        if self._text_width(draw, text, font) <= max_width:
            return text
        suffix = "..."
        cleaned = re.sub(r"\s+", " ", text).strip()
        while cleaned and self._text_width(draw, cleaned + suffix, font) > max_width:
            cleaned = cleaned[:-1].rstrip()
        return cleaned + suffix if cleaned else suffix

    def _text_width(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    ) -> int:
        box = draw.textbbox((0, 0), text, font=font)
        return int(box[2] - box[0])

    def _text_height(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    ) -> int:
        box = draw.textbbox((0, 0), text, font=font)
        return int(box[3] - box[1])

    def _hex_to_rgb(self, value: str) -> tuple[int, int, int]:
        cleaned = value.lstrip("#")
        return (int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16))
