"""Catalog sản phẩm: đọc PostgreSQL database `db_dmx` (hoặc schema chỉ định).

Hỗ trợ bảng `public.products` từ container `dmx_postgres` (42.000+ sản phẩm),
đồng thời giữ khả năng tương thích ngược.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from app.core.exceptions import DataSourceError
from app.services.group_scores import GROUP_NAMES

logger = logging.getLogger("app.catalog")

# Từ điển một sản phẩm.
Product = dict

# Ánh xạ tên danh mục từ db_dmx sang một hoặc nhiều mã nhóm group_code (62 nhóm chuẩn)
DB_CATEGORY_TO_GROUPS: dict[str, list[str]] = {
    "Tivi": ["tv"], "TV": ["tv"],
    "Máy lạnh": ["air_conditioner"], "Máy điều hòa": ["air_conditioner"],
    "Quạt": ["fan"], "Quạt mini": ["fan"], "Quạt điện": ["fan"],
    "Quạt điều hoà": ["air_cooler"], "Quạt điều hòa": ["air_cooler"],
    "Quạt trần": ["ceiling_fan"],
    "Quạt sưởi": ["heater_fan"],
    "Loa": ["speaker"],
    "Micro karaoke": ["karaoke_speaker"], "Loa karaoke": ["karaoke_speaker"],
    "Ghế massage": ["massage_chair"], "Ghế Massage": ["massage_chair"],
    "Máy lọc không khí": ["air_purifier"],
    "Robot hút bụi": ["robot_vacuum"],
    "Máy hút bụi": ["vacuum_cleaner"], "Máy hút bụi & Thiết bị làm sạch sàn": ["vacuum_cleaner"],
    "Máy chiếu": ["projector"], "Máy chiếu & Màn hình chiếu": ["projector"],
    "Đèn điện, đèn sạc": ["lamp"], "Đèn điện/đèn sạc": ["lamp"],
    "Bóng đèn led": ["led_bulb"], "Bóng đèn LED": ["led_bulb"],
    "Đèn năng lượng mặt trời": ["solar_light"],
    "Ổ cắm điện & Dây nối": ["power_strip"], "Ổ cắm, bóng đèn thông minh": ["power_strip"],
    "Ổ cắm - Phích cắm": ["power_strip"], "Ổ cắm điện": ["power_strip"],
    "Tủ lạnh": ["refrigerator"],
    "Tủ đông, tủ mát": ["freezer", "beverage_cooler"], "Tủ đông": ["freezer"], "Tủ mát": ["beverage_cooler"],
    "Máy giặt": ["washing_machine"],
    "Máy sấy quần áo": ["clothes_dryer"], "Tủ sấy quần áo": ["clothes_dryer"],
    "Máy rửa chén": ["dishwasher"],
    "Nồi chiên không dầu": ["air_fryer"],
    "Nồi cơm điện": ["rice_cooker"],
    "Lò vi sóng": ["microwave"],
    "Lò nướng": ["oven"],
    "Bếp từ": ["induction_cooker"],
    "Bếp hồng ngoại": ["infrared_cooker"],
    "Bếp gas": ["gas_stove"], "Bếp ga": ["gas_stove"],
    "Bình đun siêu tốc": ["kettle"], "Ấm đun nước": ["kettle"],
    "Cây nước nóng lạnh": ["water_dispenser"],
    "Máy nước nóng": ["water_heater"],
    "Máy lọc nước": ["water_purifier"],
    "Máy xay sinh tố": ["blender"],
    "Máy ép trái cây": ["juicer"],
    "Máy làm sữa hạt": ["soy_milk_maker"],
    "Máy pha cà phê": ["coffee_maker"],
    "Bàn ủi": ["iron"],
    "Bồn ngâm chân": ["foot_bath"],
    "Cân sức khoẻ": ["body_scale"], "Cân sức khỏe và phân tích cơ thể": ["body_scale"], "Cân sức khỏe": ["body_scale"],
    "Máy sấy tóc": ["hair_dryer"],
    "Máy tạo kiểu tóc": ["hair_styler"],
    "Máy cạo râu": ["shaver"],
    "Bàn chải điện": ["electric_toothbrush"], "Bàn chải điện & phụ kiện": ["electric_toothbrush"], "Bàn chải đánh răng": ["electric_toothbrush"], "Bàn chải đánh răng điện": ["electric_toothbrush"],
    "Máy tăm nước": ["water_flosser"],
    "Máy rửa mặt": ["facial_cleanser"],
    "Máy tỉa lông mũi": ["trimmer"], "Máy tỉa lông mũi, lông tai": ["trimmer"], "Dụng cụ tẩy lông": ["trimmer"], "Máy tỉa lông": ["trimmer"],
    "Máy massage cá nhân": ["massager"], "Dụng cụ massage và trị liệu": ["massager"], "Thiết bị massage": ["massager"],
    "Laptop": ["laptop"],
    "Máy tính để bàn": ["desktop_pc"], "Bộ Máy Tính Bàn": ["desktop_pc"],
    "Màn hình máy tính": ["monitor"],
    "Camera giám sát": ["indoor_camera", "outdoor_camera"], "Camera giám sát kết nối internet": ["indoor_camera"], "Camera trong nhà": ["indoor_camera"],
    "Camera hành trình / hành động": ["outdoor_camera"], "Camera hành trình & Camera lùi": ["outdoor_camera"], "Camera ngoài trời": ["outdoor_camera"],
    "Máy chơi game": ["game_console"],
    "Máy hút ẩm": ["dehumidifier"],
    "Máy hút mùi": ["range_hood"],
    "Bộ lau nhà": ["mop_set"], "Bộ & Cây lau nhà": ["mop_set"],
    "Thùng rác": ["trash_bin"],
    "Hộp đựng đồ đa năng": ["storage_box"], "Hộp đựng thực phẩm": ["storage_box"], "Hộp khăn giấy": ["storage_box"], "Hộp đựng đồ": ["storage_box"],
    "Bảng treo đồ Pegboard": ["storage_rack"], "Khay/kệ đựng đồ": ["storage_rack"],
    "Micro thu âm": ["microphone"], "Micro thu âm điện thoại": ["microphone"], "Microphone": ["microphone"],
}


class Catalog:
    """Gom sản phẩm theo `group_code` và giữ sẵn danh sách còn hàng."""

    def __init__(self, groups: dict[str, list[Product]], source: str):
        self.groups: dict[str, list[Product]] = groups
        self.source = source
        self.counts: dict[str, int] = {g: len(v) for g, v in groups.items()}
        # Lọc `availability` tính sẵn một lần.
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
        "brand": str(row.get("brand_name") or row.get("brand") or "Điện máy XANH").strip(),
        "price": float(row["price"]),
        "currency": str(row.get("currency") or "VND").strip(),
        "image_url": str(row.get("image_url") or "").strip(),
        "product_url": str(row.get("product_url") or "").strip(),
        "availability": bool(row["availability"]),
    }


def load_catalog_from_db(cfg: dict, schema: str = "public") -> Catalog:
    """Đọc dữ liệu sản phẩm từ PostgreSQL (`db_dmx` hoặc `mwg`)."""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError as exc:  # pragma: no cover
        raise DataSourceError(
            "Thiếu driver psycopg2 — không đọc được PostgreSQL."
        ) from exc

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

            # Trường hợp 1: Database db_dmx (bảng `products`)
            if "products" in existing:
                cur.execute(
                    f"""
                    SELECT p.product_id,
                           p."tên sản phẩm" AS product_name,
                           COALESCE(NULLIF(p.brand, ''), 'Điện máy XANH') AS brand_name,
                           COALESCE(NULLIF(p."Giá khuyến mãi", 0), NULLIF(p."Giá gốc", 0), 0) AS price,
                           'VND' AS currency,
                           COALESCE(p.url_image, '') AS image_url,
                           COALESCE(p.url, '') AS product_url,
                           (p.url_image IS NOT NULL AND length(p.url_image) > 5 AND COALESCE(p."Giá khuyến mãi", p."Giá gốc", 0) > 0) AS availability,
                           p.category_name
                    FROM {schema}.products p
                    WHERE p.url_image IS NOT NULL AND length(p.url_image) > 5 AND COALESCE(p."Giá khuyến mãi", p."Giá gốc", 0) > 0
                    ORDER BY p.product_id
                    """
                )
                rows = cur.fetchall()

                groups: dict[str, list[Product]] = defaultdict(list)
                for row in rows:
                    prod = _row_to_product(row)
                    cname = row.get("category_name")
                    target_groups = DB_CATEGORY_TO_GROUPS.get(cname, [])
                    for g in target_groups:
                        groups[g].append(prod)

                total_items = sum(len(v) for v in groups.values())
                logger.info(
                    "nạp catalog từ db_dmx: %d sản phẩm ánh xạ vào %d nhóm",
                    total_items, len(groups), extra={"event": "catalog_db"}
                )
                return Catalog(dict(groups), source=f"PostgreSQL {cfg.get('dbname')}.products ({len(rows)} sản phẩm)")

            # Trường hợp 2: Schema cũ mwg (`product` + `brand`)
            elif "product" in existing and "brand" in existing:
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
                groups = defaultdict(list)
                for row in rows:
                    groups[row["group_code"]].append(_row_to_product(row))

                return Catalog(dict(groups), source=f"PostgreSQL {schema}.product ({len(rows)} sản phẩm)")

            else:
                raise DataSourceError(
                    f"Schema '{schema}' không chứa bảng 'products' hay 'product'. Đang có: {sorted(existing)}"
                )
    finally:
        conn.close()


def load_category_names(cfg: dict, schema: str = "public") -> dict[str, str]:
    """Tên hiển thị (`TV`, `Loa`...) -> `group_code` (`tv`, `speaker`...).

    Đảm bảo đầy đủ 62 nhóm để khớp với `category_room_scores.json`.
    """
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover
        raise DataSourceError("Thiếu driver psycopg2.") from exc

    # Kiểm tra kết nối tới DB
    try:
        conn = psycopg2.connect(**cfg)
        conn.close()
    except Exception as exc:  # noqa: BLE001
        raise DataSourceError(
            f"Không kết nối được PostgreSQL ({cfg.get('host')}:{cfg.get('port')}/"
            f"{cfg.get('dbname')}): {type(exc).__name__}: {exc}"
        ) from exc

    # Ánh xạ chuẩn 62 nhóm
    mapping = {name: code for code, name in GROUP_NAMES.items()}
    return mapping
