"""Tuỳ biến OpenAPI/Swagger cho dự án.

`configure_openapi()` gắn mô tả ngắn, tag và phản hồi lỗi dùng chung vào schema
OpenAPI mà Swagger UI (`/docs`) hiển thị.

Mọi thứ ở đây **chỉ là tài liệu** — không thay đổi hành vi của endpoint.
Mô tả giữ ở mức một dòng: chi tiết công thức nằm trong README, không lặp lại ở đây.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

API_DESCRIPTION = (
    "Gợi ý sản phẩm điện máy từ ảnh phòng: Grounding DINO (Apache-2.0) + đồng xuất hiện. "
    "Không có bước train — model chạy zero-shot trên 45 class chuẩn."
)

TAGS_METADATA: list[dict[str, Any]] = [
    {"name": "recommendations", "description": "Ảnh phòng → nhóm sản phẩm gợi ý."},
    {"name": "health", "description": "Tiến trình và mức sẵn sàng."},
]

# Phản hồi lỗi dùng lại cho mọi endpoint.
ERROR_RESPONSE_REF: dict[str, Any] = {"$ref": "#/components/schemas/ErrorResponse"}


ERROR_CONTENT: dict[str, Any] = {"application/json": {"schema": ERROR_RESPONSE_REF}}


def error_response(description: str) -> dict[str, Any]:
    """Khối `responses` cho một mã lỗi, trỏ tới schema `ErrorResponse` dùng chung."""
    return {"description": description, "content": ERROR_CONTENT}


ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    400: error_response("Ảnh không hợp lệ."),
    404: error_response("Tham số không hợp lệ."),
    429: error_response("Quá nhiều yêu cầu đang xử lý."),
    503: error_response("Model hoặc dữ liệu chưa sẵn sàng."),
}


def _register_error_schemas(schema: dict[str, Any]) -> None:
    """Đưa `ErrorResponse`/`ErrorDetail` vào components để `$ref` phân giải được."""
    from app.schemas.recommendation import ErrorResponse

    components = schema.setdefault("components", {}).setdefault("schemas", {})
    if "ErrorResponse" in components:
        return

    raw = ErrorResponse.model_json_schema()
    defs = raw.pop("$defs", {})
    for name, body in defs.items():
        components.setdefault(name, body)
    components["ErrorResponse"] = raw


def _mark_multipart_images_as_binary(schema: dict[str, Any]) -> None:
    """Tương thích Swagger UI: upload mảng file cần `format: binary`.

    FastAPI/Pydantic phát OpenAPI 3.1 bằng ``contentMediaType``. Phiên bản
    Swagger UI đang dùng không render nút chọn file cho mảng ảnh nếu thiếu
    ``format: binary``, nên nó hiển thị ô nhập chuỗi/đường dẫn thay vì input
    type=file. Chỉ bổ sung metadata tài liệu, không đổi payload API.
    """
    components = schema.get("components", {}).get("schemas", {})
    for name in ("Body_recommendThreeRoomsAuto", "Body_recommendThreeRoomsByType"):
        image_items = (
            components.get(name, {})
            .get("properties", {})
            .get("images", {})
            .get("items", {})
        )
        if image_items.get("type") == "string":
            image_items["format"] = "binary"


def configure_openapi(app: FastAPI) -> None:
    """Gắn metadata + phản hồi lỗi vào schema OpenAPI. Gọi một lần sau khi tạo app."""

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            summary=getattr(app, "summary", None),
            description=app.description,
            routes=app.routes,
            tags=TAGS_METADATA,
            contact=getattr(app, "contact", None),
            license_info=getattr(app, "license_info", None),
        )

        paths: dict[str, Any] = schema.get("paths", {})
        for path, methods in paths.items():
            if not path.startswith("/api/v1/"):
                continue
            for method, operation in methods.items():
                if method not in {"get", "post", "put", "patch", "delete"}:
                    continue
                # Mỗi route đã tự khai báo mã lỗi đúng với nó; chỉ bổ sung `503` cho
                # endpoint cần model, không thêm lỗi không thể xảy ra.
                operation.setdefault("responses", {}).setdefault("503", ERROR_RESPONSES[503])

        _register_error_schemas(schema)
        _mark_multipart_images_as_binary(schema)

        # Server dùng khi chạy `uvicorn --port 8000` tại máy.
        schema.setdefault("servers", [{"url": "/", "description": "Server hiện tại"}])
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]
