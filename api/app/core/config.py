"""Cấu hình đọc từ biến môi trường / `api/.env`.

Mọi tham số ở đây ánh xạ 1-1 tới hằng số trong notebook
`yolo_train_predict_v5_spatial_relationships_rtdetr.ipynb`, cell 1 → 10.14.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]  # .../api

TOP_K_MAX = 100
PRODUCTS_PER_GROUP_MAX = 50

# Cell 8 của notebook (train) KHÔNG được port: API chỉ đọc checkpoint .pt có sẵn.
ALLOWED_IMAGE_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # --- Ứng dụng ---
    app_name: str = "Room Recommendation API (v5)"
    app_version: str = "5.0.0"
    environment: str = "development"
    log_level: str = "INFO"
    docs_enabled: bool = True

    # --- Tầng 0: nạp model Grounding DINO (Zero-Shot, Apache-2.0) ---
    # Nhận Hugging Face model ID (vd: `IDEA-Research/grounding-dino-tiny`) hoặc
    # đường dẫn tới thư mục snapshot đã lưu cục bộ.
    model_path: str = "IDEA-Research/grounding-dino-tiny"
    model_device: str = "auto"  # auto | cpu | mps | cuda | 0
    model_confidence: float = Field(0.30, ge=0, le=1)
    model_text_threshold: float = Field(0.25, ge=0, le=1)
    # Tái tạo bước nén JPEG trong bộ nhớ để chuẩn hoá ảnh RGB đầu vào.
    detect_jpeg_roundtrip: bool = True
    # Grounding DINO mặc định resize shortest edge lên 800px. Interior object
    # detection chỉ cần bố cục/vật thể lớn nên 512px giảm đáng kể CPU latency.
    detect_shortest_edge: int = Field(512, ge=320, le=800)
    detect_longest_edge: int = Field(768, ge=480, le=1333)
    # Bắt buộc bộ class chuẩn của detector khớp `cooccurrence_matrix.json`.
    require_class_match: bool = True

    # --- Cell 10.3 / 10.5: ma trận đồng xuất hiện học từ COCO train ---
    cooccurrence_matrix_path: str = "artifacts/matrix/cooccurrence_matrix.json"
    # Cell 10.7: bảng điểm W(c, r), khoá theo TÊN nhóm hiển thị.
    category_room_scores_path: str = "artifacts/matrix/category_room_scores.json"
    # Đường dẫn dữ liệu COCO — chỉ dùng để đối chiếu class của checkpoint.
    # Để trống = bỏ qua bước so class (khi không mang theo annotation).
    data_root: str = ""

    # --- Cell 10.1 / 10.6: catalog sản phẩm ---
    # Catalog đọc từ PostgreSQL database `db_dmx` (container `dmx_postgres`).
    db_host: str = "127.0.0.1"
    db_port: int = 5433
    db_name: str = "db_dmx"
    db_user: str = "postgres"
    db_password: str = "postgres"
    db_schema: str = "public"
    db_connect_timeout_s: int = Field(5, ge=1)

    # --- Tham số gợi ý (Cell 10.1) ---
    default_room_type: str = "living_room"
    default_top_k: int = Field(10, ge=1, le=TOP_K_MAX)
    default_min_score: float = Field(0.05, ge=0)
    products_per_group: int = Field(3, ge=1, le=PRODUCTS_PER_GROUP_MAX)
    default_exclude_detected: bool = True
    default_min_price: float = Field(
        250_000.0,
        ge=0,
        description="Mức giá tối thiểu mặc định (VND) cho sản phẩm gợi ý (mặc định 250k = 250.000đ).",
    )
    # Cell 10.9: hệ số Laplace smoothing.
    alpha_smooth: float = Field(1.0, ge=0)
    smooth_k: int = Field(2, ge=1)

    # --- Qwen Image Edit Gateway (ghép sản phẩm qua /v1/images/edits) ---
    qwen_gateway_url: str = Field(
        default="",
        description="URL của Qwen Image Edit gateway, đọc từ QWEN_GATEWAY_URL trong .env",
    )
    qwen_api_key: str = Field(
        default="",
        description="API key xác thực Qwen gateway, đọc từ QWEN_API_KEY trong .env",
    )
    qwen_gateway_timeout_s: float = Field(60.0, gt=0)
    qwen_gateway_connect_timeout_s: float = Field(10.0, gt=0)
    # Giới hạn dùng chung cho mọi request gọi Qwen trong cùng API process.
    # Batch ba phòng mặc định chạy hai edits song song để không quá tải gateway.
    qwen_max_concurrent_edits: int = Field(2, ge=1, le=3)
    auto_generate_image: bool = False

    # --- Giới hạn đầu vào ---
    max_upload_mb: float = Field(10, gt=0)
    max_image_pixels: int = Field(25_000_000, gt=0)
    min_image_side: int = Field(32, ge=1)
    max_pending_requests: int = Field(4, ge=1)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    def resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        return p if p.is_absolute() else (BASE_DIR / p).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
