from __future__ import annotations

import base64
import json
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path
from typing import Protocol

import httpx
from PIL import Image, ImageDraw, ImageFilter

from app.core.config import settings
from app.core.ids import stable_id
from app.models import GenerationJob


class ImageGenerationAdapter(Protocol):
    def generate(self, job: GenerationJob) -> dict[str, object]:
        """Generate or edit an image and return output metadata."""


class MockImageGenerationAdapter:
    def __init__(self, output_dir: Path = Path("data/generated")) -> None:
        self.output_dir = output_dir

    def generate(self, job: GenerationJob) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_id = stable_id("out", job.id, job.attempt)
        image_path = self.output_dir / f"{output_id}.png"
        request_path = self.output_dir / f"{output_id}.request.json"

        image = Image.new("RGB", (1024, 1024), color=(218, 222, 224))
        draw = ImageDraw.Draw(image)
        draw.rectangle((92, 420, 932, 680), outline=(68, 76, 82), width=6)
        draw.ellipse((190, 650, 310, 770), fill=(34, 38, 42))
        draw.ellipse((715, 650, 835, 770), fill=(34, 38, 42))
        image.save(image_path)

        request_path.write_text(str(job.request_json), encoding="utf-8")
        return {
            "output_id": output_id,
            "image_uri": str(image_path.resolve()),
            "width": 1024,
            "height": 1024,
            "request_uri": str(request_path.resolve()),
        }


class OpenAIImageGenerationAdapter:
    def __init__(
        self,
        output_dir: Path = Path("data/generated"),
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_image_model

    def generate(self, job: GenerationJob) -> dict[str, object]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIImageGenerationAdapter")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_id = stable_id("out", job.id, job.attempt, self.model)
        image_path = self.output_dir / f"{output_id}.png"
        request_path = self.output_dir / f"{output_id}.request.json"
        response_path = self.output_dir / f"{output_id}.response.json"
        raw_image_path = self.output_dir / f"{output_id}.raw.png"

        prompt = self._prompt_for_image_model(job)
        revision_instruction = job.request_json.get("revision_instruction")
        if revision_instruction:
            prompt = (
                f"{prompt}\n\nRevision instruction: {revision_instruction}\n"
                "This retry must visibly address every QA failure while preserving product facts."
            )

        source_path = self._source_image_path(job)
        if source_path is not None:
            endpoint = "images/edits"
            payload = self._edit_payload(prompt, source_path, job)
            request_path.write_text(
                json.dumps(
                    {
                        "endpoint": endpoint,
                        "payload": payload,
                        "source_image_uri": str(source_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            with httpx.Client(timeout=settings.openai_request_timeout_seconds) as client:
                response = self._post_edit_with_retries(client, payload, source_path)
                response.raise_for_status()
                body = response.json()
        else:
            endpoint = "images/generations"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "n": 1,
                "size": settings.openai_image_size,
                "output_format": "png",
            }
            request_path.write_text(
                json.dumps(
                    {"endpoint": endpoint, "payload": payload},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            with httpx.Client(timeout=settings.openai_request_timeout_seconds) as client:
                response = self._post_generation_with_retries(client, payload)
                response.raise_for_status()
                body = response.json()

        response_path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_image_response(body, raw_image_path)
        self._normalize_ecommerce_image(raw_image_path, image_path)

        with Image.open(image_path) as image:
            width, height = image.size

        return {
            "output_id": output_id,
            "image_uri": str(image_path.resolve()),
            "width": width,
            "height": height,
            "request_uri": str(request_path.resolve()),
            "response_uri": str(response_path.resolve()),
            "raw_image_uri": str(raw_image_path.resolve()),
            "source_image_uri": str(source_path.resolve()) if source_path is not None else None,
        }

    def _source_image_path(self, job: GenerationJob) -> Path | None:
        source_image_uri = job.request_json.get("source_image_uri")
        if not source_image_uri:
            return None
        if (
            job.route != "clean_edit"
            and job.request_json.get("generation_mode") != "source_image_edit"
            and job.route != "structure_preserve_rebuild"
            and job.request_json.get("generation_mode") != "structure_preserve_rebuild"
        ):
            return None
        source_path = Path(str(source_image_uri))
        if not source_path.exists():
            raise FileNotFoundError(f"Source image for edit does not exist: {source_path}")
        return source_path

    def _edit_payload(
        self, prompt: str, source_path: Path, job: GenerationJob | None = None
    ) -> dict[str, object]:
        return {
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": settings.openai_image_size,
            "quality": "high",
            "background": "auto",
            "output_format": "png",
            "source_filename": source_path.name,
            "source_risk_regions": job.request_json.get("source_risk_regions", [])
            if job is not None
            else [],
        }

    def _write_image_response(self, body: dict[str, object], image_path: Path) -> None:
        first: object | None = None
        data = body.get("data")
        if isinstance(data, list) and data:
            first = data[0]
        elif "b64_json" in body or "url" in body:
            first = body

        if not isinstance(first, dict):
            raise RuntimeError("OpenAI image response did not include data")

        if "b64_json" in first:
            image_path.write_bytes(base64.b64decode(str(first["b64_json"])))
        elif "url" in first:
            with httpx.Client(timeout=settings.openai_request_timeout_seconds) as client:
                image_response = client.get(str(first["url"]))
                image_response.raise_for_status()
                image_path.write_bytes(image_response.content)
        else:
            raise RuntimeError("OpenAI image response did not include b64_json or url")

    def _normalize_ecommerce_image(self, input_path: Path, output_path: Path) -> None:
        target_size = self._parse_size(settings.ecommerce_image_size)
        if target_size is None:
            input_path.replace(output_path)
            return

        target_width, target_height = target_size
        with Image.open(input_path) as source:
            image = source.convert("RGB")
        if image.size == target_size:
            image.save(output_path)
            return

        if settings.ecommerce_image_fit == "cover":
            normalized = self._cover_resize(image, target_width, target_height)
        else:
            normalized = self._contain_blur_resize(image, target_width, target_height)
        normalized.save(output_path, format="PNG")

    def _parse_size(self, value: str) -> tuple[int, int] | None:
        if not value or value.lower() == "auto":
            return None
        try:
            width_text, height_text = value.lower().split("x", maxsplit=1)
            width = int(width_text)
            height = int(height_text)
        except ValueError as exc:
            raise ValueError(f"Invalid image size: {value}") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {value}")
        return width, height

    def _contain_blur_resize(self, image: Image.Image, width: int, height: int) -> Image.Image:
        canvas = self._cover_resize(image, width, height).filter(ImageFilter.GaussianBlur(24))
        canvas = canvas.point(lambda channel: int(channel * 0.82 + 255 * 0.18))
        foreground = image.copy()
        foreground.thumbnail((width, height), Image.Resampling.LANCZOS)
        left = (width - foreground.width) // 2
        top = (height - foreground.height) // 2
        canvas.paste(foreground, (left, top))
        return canvas

    def _cover_resize(self, image: Image.Image, width: int, height: int) -> Image.Image:
        scale = max(width / image.width, height / image.height)
        resized = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
        left = max(0, (resized.width - width) // 2)
        top = max(0, (resized.height - height) // 2)
        return resized.crop((left, top, left + width, top + height))

    def _post_generation_with_retries(
        self, client: httpx.Client, payload: dict[str, object]
    ) -> httpx.Response:
        last_error: Exception | None = None
        max_retries = max(1, settings.openai_max_retries)
        for attempt in range(1, max_retries + 1):
            try:
                response = client.post(
                    f"{self.base_url}/images/generations",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= max_retries or not self._is_retryable(exc):
                    raise
                time.sleep(self._retry_delay_seconds(exc, attempt))
        raise RuntimeError("OpenAI image generation request failed") from last_error

    def _post_edit_with_retries(
        self, client: httpx.Client, payload: dict[str, object], source_path: Path
    ) -> httpx.Response:
        last_error: Exception | None = None
        max_retries = max(1, settings.openai_max_retries)
        for attempt in range(1, max_retries + 1):
            try:
                response = self._post_edit_once(client, payload, source_path)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= max_retries or not self._is_retryable(exc):
                    raise
                time.sleep(self._retry_delay_seconds(exc, attempt))
        raise RuntimeError("OpenAI image edit request failed") from last_error

    def _post_edit_once(
        self, client: httpx.Client, payload: dict[str, object], source_path: Path
    ) -> httpx.Response:
        response: httpx.Response | None = None
        upload_name, upload_bytes, mime_type = self._source_image_upload(source_path)
        mask_upload = self._source_mask_upload(source_path, self._source_risk_regions(payload))
        for candidate in self._edit_payload_fallbacks(payload):
            files = {"image": (upload_name, upload_bytes, mime_type)}
            if mask_upload is not None:
                mask_name, mask_bytes, mask_mime_type = mask_upload
                files["mask"] = (mask_name, mask_bytes, mask_mime_type)
            data = {
                key: str(value)
                for key, value in candidate.items()
                if key not in {"source_filename", "source_risk_regions"}
            }
            response = client.post(
                f"{self.base_url}/images/edits",
                headers={"Authorization": f"Bearer {self.api_key}"},
                data=data,
                files=files,
            )
            if response.status_code != 400:
                return response
        if response is None:
            raise RuntimeError("OpenAI image edit request did not send a request")
        return response

    def _source_image_upload(self, source_path: Path) -> tuple[str, bytes, str]:
        buffer = BytesIO()
        with Image.open(source_path) as image:
            image.convert("RGBA" if image.mode in {"LA", "RGBA"} else "RGB").save(
                buffer, format="PNG"
            )
        return f"{source_path.stem}.png", buffer.getvalue(), "image/png"

    def _source_mask_upload(
        self, source_path: Path, risk_regions: list[dict[str, object]]
    ) -> tuple[str, bytes, str] | None:
        if not risk_regions:
            return None
        with Image.open(source_path) as image:
            width, height = image.size
        mask = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(mask)
        for region in risk_regions:
            box = self._region_box(region, width, height)
            if box is not None:
                draw.rectangle(box, fill=(0, 0, 0, 0))
        buffer = BytesIO()
        mask.save(buffer, format="PNG")
        return f"{source_path.stem}.mask.png", buffer.getvalue(), "image/png"

    def _source_risk_regions(self, payload: dict[str, object]) -> list[dict[str, object]]:
        raw_regions = payload.get("source_risk_regions", [])
        if not isinstance(raw_regions, list):
            return []
        regions: list[dict[str, object]] = []
        for region in raw_regions:
            if isinstance(region, dict):
                regions.append(region)
        return regions

    def _region_box(
        self, region: dict[str, object], image_width: int, image_height: int
    ) -> tuple[int, int, int, int] | None:
        try:
            x = float(str(region.get("x", 0)))
            y = float(str(region.get("y", 0)))
            width = float(str(region.get("width", 0)))
            height = float(str(region.get("height", 0)))
        except ValueError:
            return None
        if width <= 0 or height <= 0:
            return None
        if max(x, y, width, height) <= 1.0:
            left = x * image_width
            top = y * image_height
            right = (x + width) * image_width
            bottom = (y + height) * image_height
        else:
            left = x
            top = y
            right = x + width
            bottom = y + height
        pad_x = max(8, int(image_width * 0.015))
        pad_y = max(8, int(image_height * 0.015))
        return (
            max(0, int(left) - pad_x),
            max(0, int(top) - pad_y),
            min(image_width, int(right) + pad_x),
            min(image_height, int(bottom) + pad_y),
        )

    def _edit_payload_fallbacks(self, payload: dict[str, object]) -> list[dict[str, object]]:
        minimal = {
            "model": payload["model"],
            "prompt": payload["prompt"],
            "n": payload["n"],
            "size": payload["size"],
            "source_risk_regions": payload.get("source_risk_regions", []),
        }
        square = dict(minimal)
        square["size"] = "1024x1024"
        return [payload, minimal, square]

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
        return isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ),
        )

    def _retry_delay_seconds(self, exc: Exception, attempt: int) -> float:
        retry_after = self._retry_after_seconds(exc)
        max_delay = max(0.0, float(settings.openai_retry_max_delay_seconds))
        if retry_after is not None:
            return float(min(max(0.0, retry_after), max_delay))
        initial_delay = max(0.0, float(settings.openai_retry_initial_delay_seconds))
        return float(min(initial_delay * (2 ** (attempt - 1)), max_delay))

    def _retry_after_seconds(self, exc: Exception) -> float | None:
        if not isinstance(exc, httpx.HTTPStatusError):
            return None
        value = exc.response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            pass
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return float((retry_at - datetime.now(UTC)).total_seconds())

    def _prompt_for_image_model(self, job: GenerationJob) -> str:
        prompt = str(job.request_json["prompt"])
        negative_prompt = str(job.request_json.get("negative_prompt") or "")
        hard_constraints = job.request_json.get("hard_constraints", [])
        qa_spec = job.request_json.get("qa_spec", {})
        if isinstance(hard_constraints, list):
            constraint_text = "\n".join(f"- {item}" for item in hard_constraints)
        else:
            constraint_text = str(hard_constraints)
        return (
            f"{self._source_edit_instruction(job)}{prompt}\n\n"
            f"{self._revision_instruction(job)}"
            f"{self._product_fact_instruction(job)}"
            f"{self._color_card_instruction(job)}"
            "Brand/model safety and hard constraints:\n"
            f"{constraint_text}\n\n"
            "Negative constraints that must not appear:\n"
            f"{negative_prompt}\n\n"
            "QA acceptance reminder:\n"
            f"{json.dumps(qa_spec, ensure_ascii=False)}"
        )

    def _revision_instruction(self, job: GenerationJob) -> str:
        revision = job.request_json.get("revision_instruction")
        if not isinstance(revision, str) or not revision.strip():
            return ""
        return (
            "Retry/rebrief instruction. This attempt must fix the previous QA failures:\n"
            f"{revision.strip()}\n\n"
        )

    def _product_fact_instruction(self, job: GenerationJob) -> str:
        product_facts = job.request_json.get("product_facts")
        color_card_review = job.request_json.get("color_card_review")
        product_text_policy = job.request_json.get("product_text_policy")
        if not isinstance(product_facts, dict) and not isinstance(color_card_review, dict):
            return ""
        if (
            isinstance(product_text_policy, dict)
            and product_text_policy.get("mode") == "layout_only_no_product_text"
        ):
            return (
                "Product-text policy: layout only. The source contains product text, but there "
                "is no reliable catalog color/material match. Create a visual-first layout with "
                "product/material visual panels only. Prefer zero visible blank panels. Use at "
                "most one restrained blank copy-safe area for later deterministic text, and make "
                "it material-textured, not an empty bordered rectangle. Fill the right side with "
                "material/roll/swatch/panel imagery. Do not create a stack of blank placeholder "
                "modules, no empty card grid, and keep blank_area_max_ratio=0.25 or lower. No "
                "visible wheels, tires, wheel arches, or center caps. "
                "Do not render or imply item code, color name, roll size, catalog code, or any "
                "readable product information.\n\n"
            )
        if (
            isinstance(product_text_policy, dict)
            and product_text_policy.get("mode") == "catalog_substitute_no_source_product_text"
        ):
            return (
                "Product-text policy: catalog substitute visuals only. The source contains "
                "product text, but its item code/color name is not an exact active-catalog match. "
                "Use the substitute color-card reference below as the visual/material basis. "
                "Do not render, imply, or preserve source item code, source color name, roll size, "
                "supplier SKU, catalog claim text, or any readable product information from the "
                "source image.\n\n"
            )
        facts_json = json.dumps(
            product_facts if isinstance(product_facts, dict) else {},
            ensure_ascii=False,
        )
        review_json = json.dumps(
            color_card_review if isinstance(color_card_review, dict) else {},
            ensure_ascii=False,
        )
        return (
            "Visible source product facts and catalog boundary:\n"
            f"{facts_json}\n"
            f"{review_json}\n"
            "Do not contradict explicit source product facts. If the catalog item code is "
            "unmatched, do not substitute a different family/finish catalog item as if it were "
            "the source product.\n\n"
        )

    def _color_card_instruction(self, job: GenerationJob) -> str:
        match = job.request_json.get("color_card_match")
        if not isinstance(match, dict):
            return ""
        item = match.get("item")
        if not isinstance(item, dict):
            return ""
        color_profile = item.get("color_profile") if isinstance(item, dict) else {}
        material_profile = item.get("material_profile") if isinstance(item, dict) else {}
        if not isinstance(color_profile, dict):
            color_profile = {}
        if not isinstance(material_profile, dict):
            material_profile = {}
        facts = {
            "item_no": item.get("item_no", ""),
            "name_zh": item.get("name_zh", ""),
            "name_en": item.get("name_en", ""),
            "series": item.get("series", ""),
            "film_type": item.get("film_type", ""),
            "material": item.get("material", ""),
            "color_family": item.get("color_family", ""),
            "finish": item.get("finish", ""),
            "product_size": item.get("product_size", ""),
            "thickness": item.get("thickness", ""),
            "match_confidence": match.get("confidence", ""),
            "match_reason": match.get("reason", ""),
            "color_profile": {
                "hex_approx": color_profile.get("hex_approx", ""),
                "median_rgb": color_profile.get("median_rgb", []),
                "lab_approx": color_profile.get("lab_approx", []),
                "dominant_hexes": color_profile.get("dominant_hexes", []),
                "confidence": color_profile.get("confidence", ""),
                "source": color_profile.get("source", ""),
            },
            "material_profile": {
                "top_layer": material_profile.get("top_layer", ""),
                "optical_stack": material_profile.get("optical_stack", []),
                "gloss_level": material_profile.get("gloss_level", ""),
                "specular_strength": material_profile.get("specular_strength", ""),
                "roughness": material_profile.get("roughness", ""),
                "metallic_flake": material_profile.get("metallic_flake", ""),
                "pearl_effect": material_profile.get("pearl_effect", ""),
                "view_angle_shift": material_profile.get("view_angle_shift", ""),
                "depth_effect": material_profile.get("depth_effect", ""),
                "reflection_behavior": material_profile.get("reflection_behavior", ""),
                "confidence": material_profile.get("confidence", ""),
            },
        }
        confidence = str(match.get("confidence") or "")
        if confidence == "exact_item":
            lead = (
                "Locked color-card reference. Use this exact real catalog item as the only color "
                "and material basis for the automotive film; do not invent unavailable colors, "
                "finishes, or product variants."
            )
        elif confidence == "nearest_color":
            lead = (
                "Substitute catalog color-card reference. The source color did not have an exact "
                "active-catalog item/color-name match, so this available catalog item is the "
                "replacement visual and material basis. Use this catalog color and finish, do not "
                "claim it is the source item number, and keep any product text out of the image."
            )
        else:
            lead = (
                "Candidate color-card reference. This is a nearest available catalog match from "
                "broad product facts, not a confirmed exact item number. Use it only as a "
                "candidate visual/material reference, do not claim the exact item number in the "
                "image, and keep the output marked for catalog review."
            )
        return (
            f"{lead}\n"
            "Material rendering instruction: "
            f"{material_profile.get('render_prompt_fragment', '')}\n"
            "Important: this is not a flat RGB fill. Render the transparent PET top layer, optical "
            "depth, finish-specific reflections, and installed vinyl-wrap behavior on curved car "
            "panels. The approximate color values are guidance from a PDF swatch crop, not a "
            "physical measurement.\n"
            "Photorealism requirement: render as real photographed automotive film, not CGI. Use "
            "natural lens depth, believable studio or workshop lighting, minor surface texture, "
            "tiny dust or handling imperfections, non-perfect film edges, and reflections that "
            "vary with panel curvature. Avoid sterile AI collage panels, uniformly sharp divider "
            "lines, plastic-sheet props, and identical highlight streaks across unrelated panels.\n"
            "Avoid: flat paint, plain single-color surface, toy-like plastic, broken reflections, "
            "and random colors outside the catalog reference.\n"
            f"{json.dumps(facts, ensure_ascii=False)}\n\n"
        )

    def _source_edit_instruction(self, job: GenerationJob) -> str:
        if (
            job.route == "structure_preserve_rebuild"
            or job.request_json.get("generation_mode") == "structure_preserve_rebuild"
        ):
            return (
                "Structure-preserve rebuild mode: use the provided source image as the base and "
                "structure reference. Preserve the source layout grid, panel count, panel "
                "positions, relative visual hierarchy, multi-angle/swatch/material panel roles, "
                "canvas balance, and source information architecture. Clean or replace risky "
                "text/logo areas without rendering readable text. Do not collapse the source "
                "into a single new hero image, and do not invent an unrelated layout.\n\n"
            )
        if (
            job.route != "clean_edit"
            and job.request_json.get("generation_mode") != "source_image_edit"
        ):
            return ""
        return (
            "Source-image edit mode: use the provided source image as the base. Preserve the same "
            "photo, crop, camera angle, perspective, vehicle geometry, film color, finish, panel "
            "gaps, glass boundaries, and lighting. Only remove or neutralize disallowed visible "
            "information in the masked or identified risky regions and make light ecommerce "
            "cleanup. Do not invent a new car or new scene.\n\n"
        )


def build_image_generation_adapter(
    output_dir: Path = Path("data/generated"),
) -> ImageGenerationAdapter:
    if settings.image_generation_provider.lower() == "openai":
        return OpenAIImageGenerationAdapter(output_dir=output_dir)
    return MockImageGenerationAdapter(output_dir=output_dir)
