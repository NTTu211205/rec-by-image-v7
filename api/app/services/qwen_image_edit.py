"""Dịch vụ kết nối Qwen-Image-Edit Gateway (OpenAPI 3.1).

Gọi các endpoint theo tài liệu `qwen-image-edit.openapi.json`:
- `POST /v1/images/edits`  : Chỉnh sửa ảnh theo prompt ("Mount product in picture, ... to picture 1")
"""

from __future__ import annotations

import base64
import logging
import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

logger = logging.getLogger("app.qwen_image_edit")

DEFAULT_GATEWAY_URL = ""
DEFAULT_API_KEY = ""
DEFAULT_EDIT_TARGET_PIXELS = 650_000
EDIT_SIZE_MULTIPLE = 32
EDIT_MIN_SIDE = 256
EDIT_MAX_SIDE = 2048

from app.services.prompt_builder import (
    DEFAULT_ROOM_EDIT_NEGATIVE_PROMPT,
    DEFAULT_ROOM_EDIT_PROMPT,
    ImageEditPromptBuilder,
    build_room_edit_negative_prompt,
    build_room_edit_prompt,
)


def optimize_image(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """Giới hạn kích thước ảnh tối đa max_side để giảm băng thông và tăng tốc GPU inference."""
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w = max(32, int(w * scale))
        new_h = max(32, int(h * scale))
        new_w = (new_w // 32) * 32 or 32
        new_h = (new_h // 32) * 32 or 32
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img


def dynamic_edit_size(
    image_input: bytes | Image.Image | str,
    target_pixels: int = DEFAULT_EDIT_TARGET_PIXELS,
) -> str | None:
    """Tạo ``WIDTHxHEIGHT`` gần 0,65MP, giữ tỷ lệ và bội số 32.

    Gateway chỉ chấp nhận cạnh 256–2048 và bội số 32. URL không mang thông tin
    kích thước nên trả ``None``; pipeline chính luôn truyền PIL Image của phòng.
    """
    try:
        if isinstance(image_input, Image.Image):
            width, height = image_input.size
        elif isinstance(image_input, bytes):
            with Image.open(BytesIO(image_input)) as image:
                width, height = image.size
        else:
            return None
    except Exception:  # noqa: BLE001
        return None

    if width <= 0 or height <= 0 or target_pixels <= 0:
        return None

    aspect = width / height
    ideal_width = math.sqrt(target_pixels * aspect)
    ideal_height = math.sqrt(target_pixels / aspect)

    def candidates(value: float) -> list[int]:
        center = round(value / EDIT_SIZE_MULTIPLE) * EDIT_SIZE_MULTIPLE
        return sorted({
            max(EDIT_MIN_SIDE, min(EDIT_MAX_SIDE, center + offset))
            for offset in (-EDIT_SIZE_MULTIPLE, 0, EDIT_SIZE_MULTIPLE)
        })

    # Ưu tiên giữ tỷ lệ (hệ số 4), sau đó mới tối thiểu sai lệch số pixel.
    best_width, best_height = min(
        (
            (candidate_width, candidate_height)
            for candidate_width in candidates(ideal_width)
            for candidate_height in candidates(ideal_height)
        ),
        key=lambda size: (
            4.0 * abs(math.log((size[0] / size[1]) / aspect))
            + abs(math.log((size[0] * size[1]) / target_pixels))
        ),
    )
    return f"{best_width}x{best_height}"


def to_data_uri(image_input: bytes | Image.Image | str, format: str = "JPEG") -> str:
    """Chuyển đổi bytes, PIL Image hoặc URL sang định dạng data URI / URL hợp lệ."""
    if isinstance(image_input, str):
        if image_input.startswith(("http://", "https://", "data:")):
            return image_input
        # Chuỗi base64 thuần
        return f"data:image/{format.lower()};base64,{image_input}"

    if isinstance(image_input, bytes):
        try:
            im = Image.open(BytesIO(image_input))
            return to_data_uri(im, format=format)
        except Exception:
            b64 = base64.b64encode(image_input).decode("ascii")
            return f"data:image/{format.lower()};base64,{b64}"

    if isinstance(image_input, Image.Image):
        buf = BytesIO()
        rgb = image_input.convert("RGB") if image_input.mode != "RGB" else image_input
        rgb = optimize_image(rgb, max_side=1024)
        rgb.save(buf, format=format, quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/{format.lower()};base64,{b64}"

    raise ValueError(f"Không hỗ trợ kiểu ảnh: {type(image_input)}")


def extract_image_from_response(payload: Any) -> str | None:
    """Trích xuất URL hoặc data URI từ cấu trúc phản hồi của gateway."""
    if not payload:
        return None
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("images", "image", "output", "result", "url"):
            val = payload.get(key)
            if isinstance(val, str) and val:
                return val
            if isinstance(val, list) and val:
                first = val[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    b64 = first.get("b64_json")
                    if b64:
                        return f"data:image/png;base64,{b64}"
                    return first.get("url")
        data = payload.get("data")
        if isinstance(data, list) and data:
            item = data[0]
            if isinstance(item, dict):
                b64 = item.get("b64_json")
                if b64:
                    return f"data:image/png;base64,{b64}"
                return item.get("url")
            if isinstance(item, str):
                return item
    return None


@dataclass
class PlaceResult:
    success: bool
    generated_image: str | None = None
    product_name: str | None = None
    placement: str | None = None
    prompt: str | None = None
    gateway_response: dict[str, Any] | None = None
    error: str | None = None


class QwenImageEditClient:
    """Client bất đồng bộ gọi Qwen-Image-Edit gateway."""

    def __init__(
        self,
        base_url: str = DEFAULT_GATEWAY_URL,
        api_key: str | None = DEFAULT_API_KEY,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
        auth_token: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.auth_token = auth_token
        self.http_client = http_client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = self.api_key or self.auth_token
        if token:
            headers["apikey"] = token
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _timeout(self, total: float | None = None) -> httpx.Timeout:
        t = total if total is not None else self.timeout
        return httpx.Timeout(t, connect=min(self.connect_timeout, t))

    async def edit_image(
        self,
        prompt: str,
        images: list[bytes | Image.Image | str],
        *,
        size: str | None = None,
        n: int = 1,
        seed: int | None = None,
        steps: int | None = 8,
        cfg: float | None = 1.0,
        negative_prompt: str = " ",
    ) -> PlaceResult:
        """Gọi POST /v1/images/edits để chỉnh sửa / ghép ảnh theo prompt."""
        if not self.base_url:
            return PlaceResult(success=False, prompt=prompt, error="Chưa cấu hình QWEN_GATEWAY_URL trong .env")
        if not (self.api_key or self.auth_token):
            return PlaceResult(success=False, prompt=prompt, error="Chưa cấu hình QWEN_API_KEY trong .env")

        try:
            uris = [to_data_uri(img) for img in images]
        except Exception as exc:
            err = f"Lỗi chuẩn hoá ảnh đầu vào cho Qwen edits: {exc}"
            logger.warning(err)
            return PlaceResult(success=False, prompt=prompt, error=err)

        payload: dict[str, Any] = {
            "prompt": prompt,
            "images": uris,
            "n": n,
            "negative_prompt": negative_prompt,
        }
        if size is not None:
            payload["size"] = size
        if seed is not None:
            payload["seed"] = seed
        if steps is not None:
            payload["steps"] = steps
        if cfg is not None:
            payload["cfg"] = cfg

        url = f"{self.base_url}/v1/images/edits"
        logger.info(
            "Gọi Qwen Image Edit /v1/images/edits: prompt=%r, images_count=%d, size=%s",
            prompt,
            len(uris),
            size or "gateway-default",
        )

        try:
            if self.http_client is not None:
                resp = await self.http_client.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self._timeout(),
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout()) as client:
                    resp = await client.post(url, json=payload, headers=self._headers())

            if resp.status_code != 200:
                err_msg = f"Gateway HTTP {resp.status_code}: {resp.text[:300]}"
                logger.warning("Qwen Gateway trả lỗi: %s", err_msg)
                return PlaceResult(success=False, prompt=prompt, error=err_msg)

            raw_json = resp.json()
            img = extract_image_from_response(raw_json)
            return PlaceResult(
                success=True,
                generated_image=img,
                prompt=prompt,
                gateway_response=raw_json if isinstance(raw_json, dict) else {"response": raw_json},
            )
        except httpx.ConnectError as exc:
            err = f"Không kết nối được Qwen Gateway tại {self.base_url}: {exc}"
            logger.warning(err)
            return PlaceResult(success=False, prompt=prompt, error=err)
        except httpx.TimeoutException as exc:
            err = f"Timeout khi chờ Qwen Gateway ({self.timeout}s): {exc}"
            logger.warning(err)
            return PlaceResult(success=False, prompt=prompt, error=err)
        except Exception as exc:  # noqa: BLE001
            err = f"Lỗi không xác định khi gọi Qwen Gateway: {type(exc).__name__}: {exc}"
            logger.error(err, exc_info=True)
            return PlaceResult(success=False, prompt=prompt, error=err)

    async def mount_products_to_room(
        self,
        room_image: bytes | Image.Image | str,
        products: list[dict[str, Any]],
        *,
        room_type: str | None = None,
        custom_prompt: str | None = None,
        negative_prompt: str | None = None,
        size: str | None = None,
        n: int = 1,
        seed: int | None = None,
        steps: int | None = 8,
        cfg: float | None = 1.0,
    ) -> PlaceResult:
        """Ghép các sản phẩm vào ảnh phòng qua endpoint /v1/images/edits theo định dạng:

        'Mount product in picture 2, picture 3, picture 4 ... to picture 1. Not changes room structure'
        Picture 1 là ảnh phòng; các Picture khác là ảnh sản phẩm.
        Hỗ trợ tối đa 2 sản phẩm (do gateway giới hạn max_images = 3).
        """
        if not products:
            return PlaceResult(success=False, error="Danh sách sản phẩm rỗng")

        # Gateway cấu hình max_images = 3 (1 phòng + tối đa 2 sản phẩm).
        selected_products = [product for product in products[:2] if product.get("image_url")]
        images_list: list[bytes | Image.Image | str] = [room_image]

        for prod in selected_products:
            img_url = prod.get("image_url")
            images_list.append(img_url)

        if len(images_list) < 2:
            return PlaceResult(
                success=False,
                error="Không có sản phẩm nào có hình ảnh hợp lệ để ghép",
            )

        prompt_builder = ImageEditPromptBuilder()

        if custom_prompt:
            final_prompt = custom_prompt
        elif selected_products:
            final_prompt = prompt_builder.build_prompt(selected_products, room_type=room_type)
        else:
            final_prompt = prompt_builder.default_prompt
        neg_prompt = (
            negative_prompt
            if negative_prompt is not None
            else prompt_builder.build_negative_prompt(selected_products)
        )
        primary_name = selected_products[0].get("product_name")
        primary_placement = selected_products[0].get("placement")

        resolved_size = size if size is not None else dynamic_edit_size(room_image)
        result = await self.edit_image(
            prompt=final_prompt,
            images=images_list,
            size=resolved_size,
            n=n,
            seed=seed,
            steps=steps,
            cfg=cfg,
            negative_prompt=neg_prompt,
        )
        result.product_name = primary_name
        result.placement = primary_placement
        return result
