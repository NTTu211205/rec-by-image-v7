"""Catalog sản phẩm: đọc PostgreSQL schema `mwg` — **chỉ** DB, không fallback CSV.

Trước đây module này có nhánh dự phòng đọc CSV khi DB không tới được. Nhánh đó
đã bị bỏ: chạy với catalog khác nguồn chuẩn mà không ai biết còn tệ hơn là
không chạy. ZEmpty DB / sai cổng / sai mật khẩu -> khởi động thất bại với thông
báo rõ ràng.

Schema đích (`db/init/01_schema.sql`, đúng 3 bảng):

    mwg.categories  (category_id = `subcategory` của CSV, 62 nhóm)
    mwg.brand       (brand_id, brand_name)
    mwg.product     (category_id -> categories, brand_id -> brand)

Đơn vị xếp hạng của recommender là `group_code`, ánh xạ thẳng sang
`mwg.product.category_id`.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from app.core.exceptions import DataSourceError

logger = logging.getLogger("app.catalog")

# Từ điển một sản phẩm.
Product = dict

_REQUIRED_TABLES = ("product", "categories", "brand")


class Catalog:
    """Gom sản phẩm theo `group_code` và giữ sẵn danh sách còn hàng."""

    def __init__(self, groups: dict[str, list[Product]], source: str):
        self.groups: dict[str, list[Product]] = groups
        self.source = source
        self.counts: dict[str, int] = {g: len(v) for g, v in groups.items()}
        # Notebook lọc `availability` mỗi lần gợi ý (Cell 10.10); tính sẵn một lần.
        self.available: dict[str, list[Product]] = {
            g: [p for p in items if p["availability"]] for g, items in groups.items()
        }

    @property
    def group_codes(self) -> list[str]:
        return sorted(self.groups)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _row_to_product(row: dict) -> Product:
    """Chuẩn hoá một dòng DB về shape mà recommender dùng."""
    return {
        "product_id": str(row["product_id"]).strip(),
        "product_name": str(row["product_name"]).strip(),
        "brand": str(row["brand_name"]).strip(),
        "price": float(row["price"]),
        "currency": str(row["currency"]).strip(),
        "image_url": str(row.get("image_url") or "").strip(),
        "product_url": str(row.get("product_url") or "").strip(),
        "availability": bool(row["availability"]),
    }


def load_catalog_from_db(cfg: dict, schema: str = "mwg") -> Catalog:
    """Đọc toàn bộ `mwg.product` join `mwg.brand`. Lỗi kết nối -> raise.

    Không còn nhánh trả `None`: gọi hàm này nghĩa là DB *phải* dùng được.
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError as exc:  # pragma: no cover - chỉ khi thiếu dependency
        raise DataSourceError(
            "Thiếu driver psycopg2 — không đọc được PostgreSQL."
        ) from exc

    # `schema` đến từ cấu hình, không phải input người dùng; chặn ký tự lạ để
    # không thể ghép SQL độc hại (identifier không tham số hoá được).
    if not schema.replace("_", "").isalnum():
        raise DataSourceError(f"DB_SCHEMA='{schema}' không hợp lệ.")

    try:
        conn = psycopg2.connect(**cfg)
    except Exception as exc:  # noqa: BLE001
        raise DataSourceError(
            f"Không kết nối được PostgreSQL ({cfg.get('host')}:{cfg.get('port')}/"
            f"{cfg.get('dbname')}): {type(exc).__name__}: {exc}"
        ) from exc

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                """,
                (schema,),
            )
            existing = {row["table_name"] for row in cur.fetchall()}
            missing = [t for t in _REQUIRED_TABLES if t not in existing]
            if missing:
                raise DataSourceError(
                    f"Schema '{schema}' thiếu bảng: {', '.join(missing)} "
                    f"(đang có: {sorted(existing) or 'rỗng'}). "
                    "Chạy `make import` để nạp lại catalog."
                )

            cur.execute(
                f"""
                SELECT p.product_id,
                       p.product_name,
                       b.brand_name,
                       p.price,
                       p.currency,
                       p.image_url,
                       p.product_url,
                       p.availability,
                       p.category_id AS group_code
                FROM {schema}.product AS p
                JOIN {schema}.brand AS b ON b.brand_id = p.brand_id
                ORDER BY p.category_id, p.price
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise DataSourceError(
            f"{schema}.product rỗng — chưa có sản phẩm nào. Chạy `make import`."
        )

    groups: dict[str, list[Product]] = defaultdict(list)
    for row in rows:
        groups[row["group_code"]].append(_row_to_product(row))

    logger.info("nạp catalog từ PostgreSQL: %d sản phẩm / %d nhóm",
                len(rows), len(groups), extra={"event": "catalog_db"})
    return Catalog(dict(groups), source=f"PostgreSQL {schema}.product ({len(rows)} sản phẩm)")


def load_category_names(cfg: dict, schema: str = "mwg") -> dict[str, str]:
    """`category_name` -> `category_id` từ `mwg.categories`.

    Cần vì `category_room_scores.json` khoá theo tên hiển thị (`"TV"`, `"Loa"`),
    còn catalog khoá theo mã (`tv`, `speaker`).
    """
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover
        raise DataSourceError("Thiếu driver psycopg2.") from exc

    try:
        conn = psycopg2.connect(**cfg)
    except Exception as exc:  # noqa: BLE001
        raise DataSourceError(
            f"Không kết nối được PostgreSQL để đọc tên nhóm: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT category_name, category_id FROM {schema}.categories"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise DataSourceError(
            f"{schema}.categories rỗng — không ghép được điểm phòng theo tên."
        )

    mapping = {str(name): str(code) for name, code in rows}
    duplicates = len(rows) - len(mapping)
    if duplicates:
        raise DataSourceError(
            f"{schema}.categories có {duplicates} tên trùng — không ghép được "
            "điểm phòng theo tên."
        )
    return mapping
