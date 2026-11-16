"""Dịch vụ kết nối Qwen-Image-Edit Gateway (OpenAPI 3.1).

Gọi các endpoint theo tài liệu `qwen-image-edit.openapi.json`:
- `POST /v1/images/edits`  : Chỉnh sửa ảnh theo prompt ("Mount product in picture, ... to picture 1")
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import httpx
from PIL import Image, ImageOps

logger = logging.getLogger("app.qwen_image_edit")

DEFAULT_GATEWAY_URL = ""
DEFAULT_API_KEY = ""
# Độ phân giải ảnh phòng đầu ra: ~0,36 MP (khoảng 704x512 hoặc 672x544)
# giúp giảm 25% số pixels GPU Qwen phải xử lý so với 0,48 MP trong khi vẫn
# giữ đủ chi tiết cho cả các thiết bị nhỏ trên bàn bếp.
DEFAULT_EDIT_TARGET_PIXELS = 360_000
DEFAULT_PRODUCT_MAX_SIDE = 256
DEFAULT_JPEG_QUALITY = 75
PRODUCT_CACHE_MAX_SIZE = 256
EDIT_SIZE_MULTIPLE = 32
EDIT_MIN_SIDE = 256
EDIT_MAX_SIDE = 2048

# Cache RAM cho ảnh sản phẩm đã tải về và scale (url, max_side) -> data URI
_PRODUCT_IMAGE_CACHE: dict[tuple[str, int], str] = {}

from app.services.prompt_builder import (
    DEFAULT_ROOM_EDIT_NEGATIVE_PROMPT,
    DEFAULT_ROOM_EDIT_PROMPT,
    ImageEditPromptBuilder,
    build_room_edit_negative_prompt,
    build_room_edit_prompt,
)


def _to_rgb(img: Image.Image) -> Image.Image:
    """Chuyển PIL Image sang RGB; nếu ảnh có nền trong suốt (RGBA/LA/P) thì lót nền trắng."""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert("RGB")


def optimize_image(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """Thu nhỏ (scale) toàn bộ ảnh theo tỷ lệ sao cho cạnh lớn nhất <= max_side (không cắt ảnh)."""
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w = max(32, round((w * scale) / 32) * 32)
        new_h = max(32, round((h * scale) / 32) * 32)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img


def optimize_product_image(img: Image.Image, target_side: int = DEFAULT_PRODUCT_MAX_SIDE) -> Image.Image:
    """Contain + đệm nền trắng về khung vuông target_side×target_side (vd: 256×256).

    Giữ nguyên 100% tỷ lệ vật lý và toàn bộ chi tiết sản phẩm (0% cắt rìa, 0% méo hình).
    """
    side = max(EDIT_MIN_SIDE, round(target_side / EDIT_SIZE_MULTIPLE) * EDIT_SIZE_MULTIPLE)
    rgb = _to_rgb(img)
    return ImageOps.pad(
        rgb,
        (side, side),
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
        centering=(0.5, 0.5),
    )


def parse_edit_size(size: str | None) -> tuple[int, int] | None:
    """Tách chuỗi ``WIDTHxHEIGHT`` thành ``(width, height)`` hợp lệ."""
    if not size or "x" not in size:
        return None
    try:
        w_str, h_str = size.lower().strip().split("x", 1)
        w, h = int(w_str), int(h_str)
        if w > 0 and h > 0:
            return w, h
    except ValueError:
        pass
    return None


def scale_image_to_size(
    image_input: bytes | Image.Image | str,
    target_size: str | None,
) -> bytes | Image.Image | str:
    """Scale toàn bộ ảnh phòng khớp 1:1 với ``target_size`` (WIDTHxHEIGHT), không cắt rìa."""
    dims = parse_edit_size(target_size)
    if dims is None:
        return image_input
    target_w, target_h = dims
    try:
        if isinstance(image_input, Image.Image):
            rgb = _to_rgb(image_input)
        elif isinstance(image_input, bytes):
            with Image.open(BytesIO(image_input)) as opened:
                rgb = _to_rgb(opened)
        else:
            return image_input
    except Exception:  # noqa: BLE001
        return image_input

    if rgb.size != (target_w, target_h):
        rgb = rgb.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return rgb


def dynamic_edit_size(
    image_input: bytes | Image.Image | str,
    target_pixels: int = DEFAULT_EDIT_TARGET_PIXELS,
) -> str | None:
    """Tạo ``WIDTHxHEIGHT`` gần 0,48MP, giữ tỷ lệ và bội số 32.

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
            for offset in range(-128, 129, EDIT_SIZE_MULTIPLE)
        })

    # Ưu tiên giữ tỷ lệ khớp sát tuyệt đối (hệ số 6), sau đó mới tối thiểu sai lệch số pixel.
    best_width, best_height = min(
        (
            (candidate_width, candidate_height)
            for candidate_width in candidates(ideal_width)
            for candidate_height in candidates(ideal_height)
        ),
        key=lambda size: (
            6.0 * abs(math.log((size[0] / size[1]) / aspect))
            + abs(math.log((size[0] * size[1]) / target_pixels))
        ),
    )
    return f"{best_width}x{best_height}"


def _encode_image_bytes(
    rgb: Image.Image,
    format: str = "JPEG",
    quality: int = DEFAULT_JPEG_QUALITY,
) -> bytes:
    """Nén ảnh tối ưu dung lượng (strip EXIF/ICC, Huffman optimize, subsampling 4:2:0)."""
    buf = BytesIO()
    fmt = format.upper()
    if fmt in ("JPEG", "JPG"):
        rgb.save(
            buf,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
            subsampling=2,
        )
    else:
        rgb.save(buf, format=fmt, quality=quality, optimize=True)
    return buf.getvalue()


def to_data_uri(
    image_input: bytes | Image.Image | str,
    format: str = "JPEG",
    max_side: int = 1024,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> str:
    """Chuyển đổi bytes, PIL Image hoặc URL sang định dạng data URI / URL hợp lệ."""
    if isinstance(image_input, str):
        if image_input.startswith(("http://", "https://", "data:")):
            return image_input
        # Chuỗi base64 thuần
        return f"data:image/{format.lower()};base64,{image_input}"

    if isinstance(image_input, bytes):
        try:
            with Image.open(BytesIO(image_input)) as im:
                return to_data_uri(_to_rgb(im), format=format, max_side=max_side, quality=quality)
        except Exception:
            b64 = base64.b64encode(image_input).decode("ascii")
            return f"data:image/{format.lower()};base64,{b64}"

    if isinstance(image_input, Image.Image):
        rgb = _to_rgb(image_input)
        rgb = optimize_image(rgb, max_side=max_side)
        raw_bytes = _encode_image_bytes(rgb, format=format, quality=quality)
        b64 = base64.b64encode(raw_bytes).decode("ascii")
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

    async def _fetch_and_scale_product_image(
        self,
        image_ref: bytes | Image.Image | str,
        max_side: int = DEFAULT_PRODUCT_MAX_SIDE,
    ) -> str:
        """Tải ảnh sản phẩm về RAM, scale thu nhỏ toàn bộ (không cắt rìa) và chuyển sang Data URI."""
        if isinstance(image_ref, (bytes, Image.Image)):
            return to_data_uri(image_ref, max_side=max_side)

        if not isinstance(image_ref, str) or not image_ref.startswith(("http://", "https://")):
            return to_data_uri(image_ref, max_side=max_side)

        cache_key = (image_ref, max_side)
        cached = _PRODUCT_IMAGE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        dl_timeout = httpx.Timeout(8.0, connect=4.0)
        try:
            if self.http_client is not None:
                resp = await self.http_client.get(
                    image_ref,
                    timeout=dl_timeout,
                    follow_redirects=True,
                )
            else:
                async with httpx.AsyncClient(timeout=dl_timeout, follow_redirects=True) as client:
                    resp = await client.get(image_ref)

            if getattr(resp, "status_code", None) == 200 and getattr(resp, "content", None):
                with Image.open(BytesIO(resp.content)) as im:
                    orig_size = im.size
                    scaled = optimize_product_image(im, target_side=max_side)
                    uri = to_data_uri(scaled, max_side=max_side)
                if len(_PRODUCT_IMAGE_CACHE) >= PRODUCT_CACHE_MAX_SIZE:
                    _PRODUCT_IMAGE_CACHE.pop(next(iter(_PRODUCT_IMAGE_CACHE)))
                _PRODUCT_IMAGE_CACHE[cache_key] = uri
                logger.info(
                    "Đã tải & scale ảnh sản phẩm (contain+pad): url=%s, orig=%sx%s -> scaled=%sx%s (%d KB)",
                    image_ref,
                    orig_size[0],
                    orig_size[1],
                    scaled.size[0],
                    scaled.size[1],
                    len(uri) // 1024,
                )
                return uri
        except Exception as exc:  # noqa: BLE001
            logger.debug("Không tải được ảnh sản phẩm %s (%s), giữ nguyên URL", image_ref, exc)

        return image_ref

    async def edit_image(
        self,
        prompt: str,
        images: list[bytes | Image.Image | str],
        *,
        size: str | None = None,
        n: int = 1,
        seed: int | None = None,
        steps: int | None = 4,
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
        steps: int | None = 4,
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
        if not selected_products:
            return PlaceResult(
                success=False,
                error="Không có sản phẩm nào có hình ảnh hợp lệ để ghép",
            )

        resolved_size = size if size is not None else dynamic_edit_size(room_image)
        # 1) Ảnh phòng (Picture 1): chỉ scale toàn bộ khung hình khớp 1:1 với resolved_size,
        # tuyệt đối không cắt (crop) mất các phần ở rìa ảnh.
        scaled_room = scale_image_to_size(room_image, resolved_size)
        # 2) Ảnh sản phẩm (Picture 2, Picture 3): tải song song về RAM và scale thu nhỏ
        # giữ nguyên tỷ lệ (không cắt mất chân/cạnh sản phẩm).
        scaled_product_uris = await asyncio.gather(
            *(
                self._fetch_and_scale_product_image(prod["image_url"])
                for prod in selected_products
            )
        )
        images_list: list[bytes | Image.Image | str] = [scaled_room, *scaled_product_uris]

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
