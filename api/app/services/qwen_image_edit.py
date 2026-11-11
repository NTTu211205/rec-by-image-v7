"""Dịch vụ kết nối Qwen-Image-Edit Gateway (OpenAPI 3.1).

Gọi các endpoint theo tài liệu `qwen-image-edit.openapi.json`:
- `GET /health`           : Kiểm tra trạng thái gateway
- `GET /config`           : Lấy cấu hình mặc định và danh sách product_types
- `POST /v1/images/edits`  : Chỉnh sửa ảnh theo prompt ("Mount product in picture, ... to picture 1")
- `POST /v1/products/place`: Ghép sản phẩm (Picture 2) vào ảnh phòng (Picture 1)
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Mapping

import httpx
from PIL import Image

logger = logging.getLogger("app.qwen_image_edit")

DEFAULT_GATEWAY_URL = ""
DEFAULT_API_KEY = ""

# Danh sách product_types chuẩn được gateway chấp thuận trong /v1/products/place
VALID_GATEWAY_PRODUCT_TYPES: frozenset[str] = frozenset({
    "air_conditioner",
    "air_purifier",
    "dryer",
    "fan",
    "freezer",
    "microwave",
    "refrigerator",
    "small_appliance",
    "speaker",
    "tv",
    "washing_machine",
    "water_purifier",
    "other",
})

# ---------------------------------------------------------------------------
# Ánh xạ 62 nhóm sản phẩm catalog -> (product_type, product_name, placement)
# chuẩn tiếng Anh để truyền vào /v1/products/place hoặc ghép prompt cho /v1/images/edits.
# ---------------------------------------------------------------------------
GROUP_PLACEMENT_PROFILES: dict[str, tuple[str, str, str]] = {
    "air_conditioner": ("air_conditioner", "air conditioner", "mounted high on the wall"),
    "air_cooler": ("fan", "air cooler", "on the floor"),
    "air_fryer": ("small_appliance", "air fryer", "on the kitchen counter"),
    "air_purifier": ("air_purifier", "air purifier", "on the floor near the corner"),
    "beverage_cooler": ("refrigerator", "beverage cooler", "on the floor"),
    "blender": ("small_appliance", "blender", "on the kitchen table"),
    "body_scale": ("other", "digital weight scale", "on the floor"),
    "ceiling_fan": ("fan", "ceiling fan", "mounted on the ceiling"),
    "clothes_dryer": ("dryer", "clothes dryer", "on the floor in the laundry area"),
    "coffee_maker": ("small_appliance", "coffee maker", "on the table or counter"),
    "dehumidifier": ("air_purifier", "dehumidifier", "on the floor"),
    "desktop_pc": ("other", "desktop computer", "on the desk"),
    "dishwasher": ("other", "dishwasher", "on the floor under the kitchen counter"),
    "electric_toothbrush": ("other", "electric toothbrush", "on the bathroom counter or shelf"),
    "facial_cleanser": ("other", "facial cleansing brush", "on the table or vanity"),
    "fan": ("fan", "standing electric fan", "on the floor"),
    "foot_bath": ("other", "foot bath massager", "on the floor"),
    "freezer": ("freezer", "chest freezer", "on the floor"),
    "game_console": ("other", "gaming console", "on the TV stand or shelf"),
    "gas_stove": ("small_appliance", "gas stove", "on the kitchen counter"),
    "hair_dryer": ("small_appliance", "hair dryer", "on the table or vanity"),
    "hair_styler": ("small_appliance", "hair styler", "on the vanity table"),
    "heater_fan": ("fan", "fan heater", "on the floor"),
    "indoor_camera": ("other", "indoor security camera", "on the shelf or mounted on the wall"),
    "induction_cooker": ("small_appliance", "induction cooker", "on the dining table or kitchen counter"),
    "infrared_cooker": ("small_appliance", "infrared cooker", "on the kitchen counter"),
    "iron": ("small_appliance", "clothes steam iron", "on the ironing board or table"),
    "juicer": ("small_appliance", "fruit juicer", "on the kitchen counter"),
    "karaoke_speaker": ("speaker", "karaoke speaker", "on the floor"),
    "kettle": ("small_appliance", "electric kettle", "on the table or kitchen counter"),
    "lamp": ("other", "desk lamp", "on the bedside table or desk"),
    "laptop": ("other", "laptop", "on the desk or table"),
    "led_bulb": ("other", "LED light bulb", "mounted on the ceiling or lamp socket"),
    "massage_chair": ("other", "massage chair", "on the floor"),
    "massager": ("other", "massage device", "on the sofa or table"),
    "microphone": ("speaker", "wireless microphone", "on the table or TV stand"),
    "microwave": ("microwave", "microwave oven", "on the kitchen counter"),
    "monitor": ("other", "computer monitor", "on the desk"),
    "mop_set": ("other", "mop and bucket set", "on the floor in the corner"),
    "outdoor_camera": ("other", "outdoor camera", "mounted on the outdoor wall"),
    "oven": ("microwave", "electric oven", "on the kitchen counter"),
    "power_strip": ("other", "power surge protector strip", "on the floor near the wall"),
    "projector": ("other", "home projector", "on the table or shelf"),
    "range_hood": ("other", "kitchen range hood", "mounted on the wall above the stove"),
    "refrigerator": ("refrigerator", "refrigerator", "on the floor beside the kitchen cabinet (can be partially cropped at frame edge)"),
    "rice_cooker": ("small_appliance", "electric rice cooker", "on the kitchen counter"),
    "robot_vacuum": ("small_appliance", "robot vacuum cleaner", "on the floor"),
    "shaver": ("small_appliance", "electric shaver", "on the bathroom shelf or table"),
    "solar_light": ("other", "solar wall light", "mounted on the wall"),
    "soy_milk_maker": ("small_appliance", "soy milk maker", "on the kitchen counter"),
    "speaker": ("speaker", "audio speaker", "on the table, shelf or TV stand"),
    "storage_box": ("other", "storage box", "on the shelf or floor"),
    "storage_rack": ("other", "storage rack", "on the floor or tabletop"),
    "trash_bin": ("other", "waste trash bin", "on the floor in the corner"),
    "trimmer": ("small_appliance", "trimmer", "on the shelf or table"),
    "tv": ("tv", "television", "on the TV stand or shelf if it has legs or a stand, otherwise mounted flush on the wall"),
    "vacuum_cleaner": ("small_appliance", "vacuum cleaner", "on the floor"),
    "washing_machine": ("washing_machine", "washing machine", "on the floor"),
    "water_dispenser": ("water_purifier", "water dispenser", "on the floor"),
    "water_flosser": ("other", "water dental flosser", "on the bathroom counter"),
    "water_heater": ("other", "water heater", "mounted on the bathroom wall"),
    "water_purifier": ("water_purifier", "water purifier", "on the kitchen counter or floor"),
}

from app.services.prompt_builder import (
    DEFAULT_ROOM_EDIT_NEGATIVE_PROMPT,
    DEFAULT_ROOM_EDIT_PROMPT,
    ImageEditPromptBuilder,
    ProductPlacementSpec,
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
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.auth_token = auth_token

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

    async def get_health(self) -> dict[str, Any]:
        """Gọi GET /health."""
        async with httpx.AsyncClient(timeout=self._timeout(min(5.0, self.timeout))) as client:
            resp = await client.get(f"{self.base_url}/health", headers=self._headers())
            resp.raise_for_status()
            return resp.json()

    async def get_config(self) -> dict[str, Any]:
        """Gọi GET /config lấy danh sách product_types và thông số mặc định."""
        async with httpx.AsyncClient(timeout=self._timeout(min(10.0, self.timeout))) as client:
            resp = await client.get(f"{self.base_url}/config", headers=self._headers())
            resp.raise_for_status()
            return resp.json()

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
        logger.info("Gọi Qwen Image Edit /v1/images/edits: prompt=%r, images_count=%d", prompt, len(uris))

        try:
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

        # Gateway cấu hình max_images = 3 (1 phòng + tối đa 2 sản phẩm)
        selected_products = products[:2]
        images_list: list[bytes | Image.Image | str] = [room_image]
        pic_refs: list[str] = []

        for idx, prod in enumerate(selected_products, start=2):
            img_url = prod.get("image_url")
            if not img_url:
                continue
            images_list.append(img_url)
            pic_refs.append(f"picture {idx}")

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

        result = await self.edit_image(
            prompt=final_prompt,
            images=images_list,
            size=size,
            n=n,
            seed=seed,
            steps=steps,
            cfg=cfg,
            negative_prompt=neg_prompt,
        )
        result.product_name = primary_name
        result.placement = primary_placement
        return result

    async def place_product(
        self,
        room_image: bytes | Image.Image | str,
        product_image: bytes | Image.Image | str,
        *,
        group_code: str | None = None,
        product_name: str | None = None,
        placement: str | None = None,
        product_type: str | None = None,
        trim_product_border: bool = True,
        size: str | None = None,
        n: int = 1,
        seed: int | None = None,
        steps: int | None = 8,
        cfg: float | None = 1.0,
    ) -> PlaceResult:
        """Gọi POST /v1/products/place để ghép sản phẩm vào ảnh phòng."""
        # 1. Xác định product_type, product_name, placement từ bảng ánh xạ
        profile_type, profile_name, profile_place = ("other", "home appliance", "on the floor")
        if group_code and group_code in GROUP_PLACEMENT_PROFILES:
            profile_type, profile_name, profile_place = GROUP_PLACEMENT_PROFILES[group_code]

        final_product_type = product_type or profile_type
        if final_product_type not in VALID_GATEWAY_PRODUCT_TYPES:
            final_product_type = "other"

        final_product_name = product_name or profile_name
        final_placement = placement or profile_place

        if not self.base_url:
            return PlaceResult(
                success=False,
                product_name=final_product_name,
                placement=final_placement,
                error="Chưa cấu hình QWEN_GATEWAY_URL trong .env",
            )
        if not (self.api_key or self.auth_token):
            return PlaceResult(
                success=False,
                product_name=final_product_name,
                placement=final_placement,
                error="Chưa cấu hình QWEN_API_KEY trong .env",
            )

        # 2. Chuẩn hoá ảnh phòng và ảnh sản phẩm sang format gateway hỗ trợ
        try:
            room_uri = to_data_uri(room_image)
            product_uri = to_data_uri(product_image)
        except Exception as exc:
            logger.warning("Lỗi chuẩn hoá ảnh đầu vào cho Qwen gateway: %s", exc)
            return PlaceResult(
                success=False,
                product_name=final_product_name,
                placement=final_placement,
                error=f"Lỗi chuẩn hoá ảnh: {exc}",
            )

        payload: dict[str, Any] = {
            "room": room_uri,
            "product": product_uri,
            "product_type": final_product_type,
            "product_name": final_product_name,
            "placement": final_placement,
            "trim_product_border": trim_product_border,
            "n": n,
        }
        if size is not None:
            payload["size"] = size
        if seed is not None:
            payload["seed"] = seed
        if steps is not None:
            payload["steps"] = steps
        if cfg is not None:
            payload["cfg"] = cfg

        url = f"{self.base_url}/v1/products/place"
        logger.info(
            "Gọi Qwen Image Edit /v1/products/place: product=%s, placement=%s",
            final_product_name,
            final_placement,
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout()) as client:
                resp = await client.post(url, json=payload, headers=self._headers())
                if resp.status_code != 200:
                    err_msg = f"Gateway HTTP {resp.status_code}: {resp.text[:300]}"
                    logger.warning("Qwen Gateway trả lỗi: %s", err_msg)
                    return PlaceResult(
                        success=False,
                        product_name=final_product_name,
                        placement=final_placement,
                        error=err_msg,
                    )
                raw_json = resp.json()
                img = extract_image_from_response(raw_json)
                return PlaceResult(
                    success=True,
                    generated_image=img,
                    product_name=final_product_name,
                    placement=final_placement,
                    gateway_response=raw_json if isinstance(raw_json, dict) else {"response": raw_json},
                )
        except httpx.ConnectError as exc:
            err = f"Không kết nối được Qwen Gateway tại {self.base_url}: {exc}"
            logger.warning(err)
            return PlaceResult(
                success=False,
                product_name=final_product_name,
                placement=final_placement,
                error=err,
            )
        except httpx.TimeoutException as exc:
            err = f"Timeout khi chờ Qwen Gateway ({self.timeout}s): {exc}"
            logger.warning(err)
            return PlaceResult(
                success=False,
                product_name=final_product_name,
                placement=final_placement,
                error=err,
            )
        except Exception as exc:  # noqa: BLE001
            err = f"Lỗi không xác định khi gọi Qwen Gateway: {type(exc).__name__}: {exc}"
            logger.error(err, exc_info=True)
            return PlaceResult(
                success=False,
                product_name=final_product_name,
                placement=final_placement,
                error=err,
            )
