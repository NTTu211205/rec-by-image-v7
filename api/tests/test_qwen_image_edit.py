"""Unit test cho Qwen-Image-Edit gateway client & tích hợp tạo ảnh."""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.main import create_app
from app.api.routes.recommendations import _generate_recommendation_image
from app.schemas.recommendation import RecommendationResponse
from app.services.qwen_image_edit import (
    PlaceResult,
    QwenImageEditClient,
    build_room_edit_negative_prompt,
    build_room_edit_prompt,
    dynamic_edit_size,
    extract_image_from_response,
    to_data_uri,
)


class FakeResponse:
    def __init__(self, data: dict, status_code: int = 200, text: str = "OK"):
        self._data = data
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return self._data


def _jpeg(size: tuple[int, int] = (64, 64)) -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 200, 200)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_to_data_uri() -> None:
    # URL giữ nguyên
    assert to_data_uri("https://example.com/a.jpg") == "https://example.com/a.jpg"
    assert to_data_uri("data:image/png;base64,123") == "data:image/png;base64,123"

    # Bytes -> data:image/jpeg;base64,...
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buf, format="JPEG")
    raw = buf.getvalue()
    uri = to_data_uri(raw)
    assert uri.startswith("data:image/jpeg;base64,")

    # PIL Image -> data:image/jpeg;base64,...
    img = Image.new("RGB", (32, 32), color="red")
    uri_img = to_data_uri(img)
    assert uri_img.startswith("data:image/jpeg;base64,")


def test_dynamic_edit_size_keeps_aspect_near_030_megapixel() -> None:
    assert dynamic_edit_size(Image.new("RGB", (900, 628))) == "640x448"
    assert dynamic_edit_size(Image.new("RGB", (960, 730))) == "640x480"
    assert dynamic_edit_size(Image.new("RGB", (700, 478))) == "672x448"
    assert dynamic_edit_size("https://example.com/room.jpg") is None

    for size in ("640x448", "640x480", "672x448"):
        width, height = (int(value) for value in size.split("x"))
        assert width % 32 == 0 and height % 32 == 0
        assert 280_000 <= width * height <= 320_000


def test_extract_image_from_response() -> None:
    # Trường hợp images: [...]
    assert (
        extract_image_from_response({"images": ["http://test.com/out.png"]})
        == "http://test.com/out.png"
    )
    # Trường hợp data: [{"b64_json": "..."}]
    assert (
        extract_image_from_response({"data": [{"b64_json": "YWJj"}]})
        == "data:image/png;base64,YWJj"
    )
    # Trường hợp data: [{"url": "..."}]
    assert (
        extract_image_from_response({"data": [{"url": "http://test.com/d.jpg"}]})
        == "http://test.com/d.jpg"
    )
    # Chuỗi trực tiếp
    assert extract_image_from_response("http://direct.jpg") == "http://direct.jpg"
    # Rỗng
    assert extract_image_from_response({}) is None


def test_qwen_mount_products_to_room_success() -> None:
    """Kiểm tra phương thức mount_products_to_room sinh prompt 'Mount product in picture, ... to picture 1'."""
    async def _run() -> None:
        client = QwenImageEditClient("https://ai.itmwg.vn/ai-mwg/qwen-edit-image", api_key="test-key")
        mock_post = AsyncMock(
            return_value=FakeResponse(
                {"data": [{"b64_json": "composed_b64"}]}, status_code=200
            )
        )

        with patch("httpx.AsyncClient.post", mock_post):
            res = await client.mount_products_to_room(
                room_image=Image.new("RGB", (900, 628), color="white"),
                products=[
                    {
                        "product_name": "Samsung Smart TV",
                        "image_url": "https://cdn.example.com/tv.jpg",
                        "placement": "on the wall",
                    },
                    {
                        "product_name": "Karaoke Speaker",
                        "image_url": "https://cdn.example.com/speaker.jpg",
                        "placement": "on the floor",
                    },
                ],
            )
            assert res.success is True
            assert res.generated_image == "data:image/png;base64,composed_b64"
            assert "Mount products in picture 2 and picture 3 to picture 1" in (res.prompt or "")

            call_kwargs = mock_post.call_args.kwargs
            payload = call_kwargs["json"]
            assert len(payload["images"]) == 3
            assert payload["images"][1] == "https://cdn.example.com/tv.jpg"
            assert payload["images"][2] == "https://cdn.example.com/speaker.jpg"
            assert payload["size"] == "640x448"
            assert payload["steps"] == 8
            assert payload["cfg"] == 1.0
            assert mock_post.call_args.args[0].endswith("/v1/images/edits")

            headers = call_kwargs["headers"]
            assert headers["apikey"] == "test-key"
            assert headers["Authorization"] == "Bearer test-key"

    asyncio.run(_run())


def test_qwen_reuses_injected_async_client() -> None:
    async def _run() -> None:
        shared_client = AsyncMock(spec=httpx.AsyncClient)
        shared_client.post.return_value = FakeResponse(
            {"data": [{"b64_json": "composed_b64"}]}
        )
        client = QwenImageEditClient(
            "https://ai.itmwg.vn/ai-mwg/qwen-edit-image",
            api_key="test-key",
            http_client=shared_client,
        )

        result = await client.edit_image(
            "keep the room unchanged",
            [Image.new("RGB", (640, 480), color="white")],
            size="800x608",
        )

        assert result.success is True
        shared_client.post.assert_awaited_once()
        call = shared_client.post.call_args
        assert call.kwargs["json"]["size"] == "800x608"
        assert isinstance(call.kwargs["timeout"], httpx.Timeout)

    asyncio.run(_run())


def test_qwen_mount_refrigerator_and_rice_cooker_uses_one_edit() -> None:
    """Cả hai sản phẩm được gửi trong một request images/edits duy nhất."""
    async def _run() -> None:
        client = QwenImageEditClient("https://ai.itmwg.vn", api_key="test-key")
        mock_post = AsyncMock(
            return_value=FakeResponse({"data": [{"b64_json": "composed_b64"}]})
        )

        with patch("httpx.AsyncClient.post", mock_post):
            result = await client.mount_products_to_room(
                room_image=b"room_bytes",
                room_type="kitchen",
                products=[
                    {
                        "group_code": "refrigerator",
                        "product_name": "Tên thương mại không dùng trong prompt",
                        "image_url": "https://cdn.example.com/refrigerator.jpg",
                    },
                    {
                        "group_code": "rice_cooker",
                        "image_url": "https://cdn.example.com/rice-cooker.jpg",
                    },
                ],
            )

        assert result.success is True
        assert "exactly one refrigerator from picture 2" in (result.prompt or "")
        assert "exactly one compact rice cooker from picture 3" in (result.prompt or "")
        assert "Tên thương mại" not in (result.prompt or "")
        assert mock_post.await_count == 1
        payload = mock_post.call_args.kwargs["json"]
        assert len(payload["images"]) == 3
        assert payload["images"][1] == "https://cdn.example.com/refrigerator.jpg"
        assert payload["images"][2] == "https://cdn.example.com/rice-cooker.jpg"
        assert mock_post.call_args.args[0].endswith("/v1/images/edits")

    asyncio.run(_run())


def test_qwen_missing_env_config() -> None:
    """Khi thiếu QWEN_GATEWAY_URL hoặc QWEN_API_KEY trong .env thì phải báo lỗi rõ ràng."""
    async def _run() -> None:
        no_key_client = QwenImageEditClient("https://ai.itmwg.vn", api_key="")
        res1 = await no_key_client.edit_image("prompt", [b"img"])
        assert res1.success is False
        assert "QWEN_API_KEY" in (res1.error or "")

        no_url_client = QwenImageEditClient("", api_key="key")
        res2 = await no_url_client.edit_image("prompt", [b"img"])
        assert res2.success is False
        assert "QWEN_GATEWAY_URL" in (res2.error or "")

    asyncio.run(_run())


def test_qwen_batch_generation_runs_in_parallel_with_shared_limit() -> None:
    """Ba kết quả được gather đồng thời, nhưng gateway chỉ có hai edit đang chạy."""
    async def _run() -> None:
        active = 0
        peak_active = 0

        async def fake_mount(_self, **_kwargs) -> PlaceResult:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return PlaceResult(
                success=True,
                generated_image="https://cdn.example.com/composed.jpg",
                prompt="batch test prompt",
            )

        def recommendation(index: int) -> RecommendationResponse:
            return RecommendationResponse(
                room_type="living_room",
                room_type_source="provided",
                room_detection_scores=None,
                detected_objects=[],
                recommendations=[{
                    "group_code": "tv",
                    "group_name": "TV",
                    "category_id": "37",
                    "group_category_id": "electronics",
                    "cooccurrence_score": 0.5,
                    "products": [{
                        "product_id": f"P{index}",
                        "category_id": "37",
                        "category_name": "Tivi",
                        "product_name": "TV",
                        "brand": "Test",
                        "price": 1_000_000,
                        "currency": "VND",
                        "image_url": "https://cdn.example.com/product.jpg",
                        "product_url": "https://example.com/product",
                    }],
                }],
                generation_status="skipped",
            )

        state = SimpleNamespace(qwen_edit_semaphore=asyncio.Semaphore(2))
        settings = Settings(
            qwen_gateway_url="https://gateway.example.com",
            qwen_api_key="test-key",
            qwen_max_concurrent_edits=2,
        )
        with patch.object(QwenImageEditClient, "mount_products_to_room", fake_mount):
            results = await asyncio.gather(*[
                _generate_recommendation_image(
                    state=state,
                    settings=settings,
                    image=Image.new("RGB", (64, 64)),
                    recommendation=recommendation(index),
                )
                for index in range(3)
            ])

        assert peak_active == 2
        assert all(result.generation_status == "success" for result in results)

    asyncio.run(_run())


def test_recommend_endpoint_with_generate_image_mock() -> None:
    """Kiểm tra gọi /api/v1/recommendations/image kèm cờ `generate_image=true`."""
    app = create_app(Settings())
    mock_post = AsyncMock(
        return_value=FakeResponse(
            {"data": [{"b64_json": "placed_mock_b64"}]}, status_code=200
        )
    )

    with patch("httpx.AsyncClient.post", mock_post), TestClient(app) as client:
        r = client.post(
            "/api/v1/recommendations/image",
            files={"image": ("room.jpg", _jpeg(), "image/jpeg")},
            params={"room_type": "living_room", "top_k": 3, "generate_image": "true"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["generation_status"] == "success"
        assert body["generated_image"] == "data:image/png;base64,placed_mock_b64"
        assert "Do not change room structure" in body["generation_prompt"]
        # Chỉ có 1 image duy nhất ở cuối cùng (response root), không nằm trong từng nhóm
        assert "generated_image" not in body["recommendations"][0]


def test_recommend_endpoint_with_custom_prompt_mock() -> None:
    """Kiểm tra gọi /api/v1/recommendations/image với custom prompt."""
    app = create_app(Settings())
    mock_post = AsyncMock(
        return_value=FakeResponse(
            {"data": [{"b64_json": "custom_mock_b64"}]}, status_code=200
        )
    )

    with patch("httpx.AsyncClient.post", mock_post), TestClient(app) as client:
        custom_p = "Mount product in picture 2 to picture 1 above the fireplace"
        r = client.post(
            "/api/v1/recommendations/image",
            files={"image": ("room.jpg", _jpeg(), "image/jpeg")},
            params={
                "room_type": "living_room",
                "top_k": 3,
                "generate_image": "true",
                "edit_prompt": custom_p,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["generation_status"] == "success"
        assert body["generation_prompt"] == custom_p


def test_recommend_endpoint_with_generate_image_skipped_by_default() -> None:
    """Mặc định `generate_image` là False/None -> trạng thái `skipped`."""
    app = create_app(Settings())
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/recommendations/image",
            files={"image": ("room.jpg", _jpeg(), "image/jpeg")},
            params={"room_type": "living_room", "top_k": 3},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["generation_status"] == "skipped"
        assert body["generated_image"] is None


def test_build_room_edit_prompt_without_tv() -> None:
    """Khi danh sách gợi ý KHÔNG có TV thì prompt tuyệt đối không được gán Tivi."""
    prods = [
        {
            "group_code": "refrigerator",
            "product_name": "Tủ lạnh Panasonic Inverter 322 lít",
            "placement": "on the floor against the wall",
        },
        {
            "group_code": "washing_machine",
            "product_name": "Máy giặt LG Inverter 9 kg",
            "placement": "on the floor",
        },
    ]
    prompt = build_room_edit_prompt(prods)
    prompt_lower = prompt.lower()
    # Không được có tivi hay television hay tv
    assert "television" not in prompt_lower
    assert "tivi" not in prompt_lower
    assert " tv " not in prompt_lower
    # Prompt chỉ dùng category English, không đưa tên thương mại vào gateway.
    assert "Picture 2 (refrigerator)" in prompt
    assert "Picture 3 (washing machine)" in prompt
    assert "Tủ lạnh Panasonic" not in prompt
    assert "Máy giặt LG" not in prompt
    assert "Do not change room structure" in prompt

    # Negative prompt cũng không được chứa 'tv'
    neg_prompt = build_room_edit_negative_prompt(prods)
    assert "tv" not in neg_prompt.lower()


def test_build_room_edit_prompt_with_tv() -> None:
    """Khi danh sách gợi ý CÓ TV thì prompt phải chỉ định đặt TV trên kệ, không treo tường nếu có chân."""
    prods = [
        {
            "group_code": "tv",
            "product_name": "Samsung Smart TV UA32H5000F",
            "placement": "on the TV stand",
        },
        {
            "group_code": "air_conditioner",
            "product_name": "Máy lạnh Livotec Inverter 1 HP",
            "placement": "mounted high on the wall",
        },
    ]
    prompt = build_room_edit_prompt(prods)
    assert "Television" in prompt or "television" in prompt.lower()
    assert "shelf or cabinet" in prompt.lower()
    assert "MUST NOT be mounted or hung on the wall" in prompt
    assert "Do not change room structure" in prompt

    # Negative prompt phải chứa quy tắc chặn treo TV có chân
    neg_prompt = build_room_edit_negative_prompt(prods)
    assert "wall mounted tv with legs" in neg_prompt.lower()

