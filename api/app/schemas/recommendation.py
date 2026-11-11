"""Hợp đồng dữ liệu của API (Pydantic) — kèm ví dụ cho Swagger UI.

Mọi `model_config` ở đây chỉ phục vụ tài liệu (`/docs`, `/redoc`):
Swagger UI dùng chúng để điền sẵn request body và hiển thị response mẫu.
Mô tả field giữ ngắn (một cụm từ); công thức chi tiết nằm trong README.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# --- Ví dụ dùng lại nhiều nơi ---

_EXAMPLE_PRODUCT = {
    "product_id": "DMX_00D536F1",
    "product_name": "Samsung Smart TV UA32H5000F",
    "brand": "Samsung",
    "price": 4090000.0,
    "currency": "VND",
    "image_url": "https://cdn.tgdd.vn/2026/10/timerseo/361163-600x600-8.jpg",
    "product_url": "https://www.dienmayxanh.com/tivi/smart-tivi-samsung-32-inch-ua32h5000f",
}

_EXAMPLE_DETECTED = [
    {"label": "Sofa", "confidence": 0.96},
    {"label": "Rug", "confidence": 0.88},
    {"label": "TV", "confidence": 0.81},
    {"label": "Coffee Table", "confidence": 0.74},
]

_EXAMPLE_RESPONSE = {
    "room_type": "living_room",
    "room_type_source": "inferred",
    "room_detection_scores": {
        "living_room": 4.16,
        "bedroom": 0.0,
        "kitchen": 0.0,
    },
    "detected_objects": _EXAMPLE_DETECTED,
    "recommendations": [
        {
            "group_code": "speaker",
            "group_name": "Loa",
            "category_id": "electronics",
            "cooccurrence_score": 0.4123,
            "products": [_EXAMPLE_PRODUCT],
        }
    ],
    "generated_image": "https://cdn.example.com/composed_room.jpg",
    "generation_status": "success",
    "generation_error": None,
    "generation_prompt": "Mount product in picture 2, picture 3 to picture 1. Not changes room structure",
}


class Product(BaseModel):
    """Một sản phẩm lấy từ catalog (Cell 10.8)."""

    model_config = ConfigDict(json_schema_extra={"example": _EXAMPLE_PRODUCT})

    product_id: str = Field(..., description="Mã sản phẩm", examples=["DMX_00D536F1"])
    product_name: str = Field(..., description="Tên sản phẩm", examples=["Samsung Smart TV UA32H5000F"])
    brand: str = Field(..., description="Thương hiệu", examples=["Samsung"])
    price: float = Field(..., description="Giá niêm yết", examples=[4090000.0])
    currency: str = Field(..., description="Đơn vị tiền tệ", examples=["VND"])
    image_url: str = Field(..., description="Ảnh sản phẩm")
    product_url: str = Field(..., description="Trang sản phẩm")


class RecommendationItem(BaseModel):
    """Một nhóm sản phẩm được gợi ý, kèm điểm cooccurrence và danh sách sản phẩm (Cell 10.10)."""

    group_code: str = Field(..., description="Mã nhóm (= subcategory)", examples=["tv"])
    group_name: str = Field(..., description="Tên nhóm hiển thị", examples=["TV"])
    category_id: str | None = Field(
        None, description="Category chứa nhóm", examples=["electronics"]
    )
    cooccurrence_score: float = Field(
        ..., description="`P_cooc(c | D)` — Noisy-OR (Cell 10.9).", examples=[0.4123]
    )
    products: list[Product] = Field(..., description="Sản phẩm còn hàng.")


class DetectedObject(BaseModel):
    """Vật thể detector phát hiện được; nhãn đã gộp trùng giữ confidence cao nhất."""

    label: str = Field(..., description="Tên class detector", examples=["Sofa"])
    confidence: float = Field(..., ge=0, le=1, description="Độ tin cậy [0, 1]", examples=[0.96])


class RecommendationResponse(BaseModel):
    """Kết quả của `recommend()` (Cell 10.10)."""

    model_config = ConfigDict(json_schema_extra={"example": _EXAMPLE_RESPONSE})

    room_type: str = Field(..., description="Phòng đã dùng.", examples=["living_room"])
    room_type_source: str = Field(
        ..., description="Nguồn xác định phòng: provided, inferred hoặc default.", examples=["inferred"]
    )
    room_detection_scores: dict[str, float] | None = Field(
        None,
        description="Điểm suy luận theo phòng khách, phòng ngủ và nhà bếp; null khi room_type được chỉ định.",
        examples=[{"living_room": 3.2, "bedroom": 0.5, "kitchen": 0.0}],
    )
    detected_objects: list[DetectedObject] = Field(
        ..., description="Vật thể đưa vào chấm điểm."
    )
    recommendations: list[RecommendationItem] = Field(
        ..., description="Đã sắp xếp thứ tự ưu tiên gợi ý."
    )
    generated_image: str | None = Field(
        None,
        description="Ảnh phòng sau khi ghép sản phẩm hàng đầu (URL hoặc data URI).",
        examples=["https://cdn.example.com/composed_room.jpg"],
    )
    generation_status: str | None = Field(
        None,
        description="Trạng thái tạo ảnh từ Qwen Gateway.",
        examples=["success"],
    )
    generation_error: str | None = Field(
        None,
        description="Chi tiết lỗi nếu tạo ảnh thất bại.",
    )
    generation_prompt: str | None = Field(
        None,
        description="Câu prompt đã dùng để chỉnh sửa ảnh qua /v1/images/edits.",
        examples=["Mount product in picture 2, picture 3 to picture 1. Not changes room structure"],
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ready",
                "ready": True,
                "model_family": "grounding_dino",
                "detector_classes": 45,
                "model_source": "IDEA-Research/grounding-dino-tiny (grounding_dino)",
                "room_scores_source": "artifact category_room_scores.json (62 nhóm × 3 phòng) + mwg.categories",
                "cooccurrence_source": "cooccurrence_matrix.json (45 class, 1692 ảnh train)",
                "catalog_groups": 62,
                "catalog_source": "PostgreSQL mwg.product (1413 sản phẩm)",
                "startup_errors": [],
            }
        }
    )

    status: str = Field(..., description="`ready` hoặc `not_ready`.")
    ready: bool = Field(..., description="Đã nạp xong chưa.")
    model_family: str | None = Field(None, description="Họ model detector (`grounding_dino`).")
    detector_classes: int = Field(0, description="Số class chuẩn của detector.")
    model_source: str | None = Field(None, description="Nguồn model detector.")
    room_scores_source: str | None = Field(
        None, description="Nguồn `W(c, r)`."
    )
    cooccurrence_source: str | None = Field(
        None, description="Nguồn ma trận đồng xuất hiện."
    )
    catalog_groups: int = Field(0, description="Số nhóm trong catalog.")
    catalog_source: str | None = Field(None, description="Nguồn catalog sản phẩm.")
    startup_errors: list[str] = Field(
        default_factory=list, description="Lỗi không nghiêm trọng khi khởi động."
    )


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Mã lỗi.", examples=["invalid_image"])
    message: str = Field(..., description="Thông báo.")


class ErrorResponse(BaseModel):
    """Mọi lỗi nghiệp vụ đều trả cùng một phong bì."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"error": {"code": "invalid_image", "message": "Dữ liệu không phải ảnh hợp lệ."}}
        }
    )

    error: ErrorDetail = Field(..., description="Chi tiết lỗi.")
