#!/usr/bin/env python3
"""Xoá sạch rồi nạp `data/dienmayxanh_products.csv` vào PostgreSQL (schema `mwg`).

    python3 scripts/import_catalog.py               # xoá sạch rồi nạp lại
    python3 scripts/import_catalog.py --dry-run     # chỉ đọc CSV, in thống kê
    python3 scripts/import_catalog.py --dsn postgresql://user:pass@host:port/db

Một transaction: lỗi giữa đường thì DB giữ nguyên dữ liệu cũ.

Ánh xạ CSV -> 3 bảng (xem `db/init/01_schema.sql`):

    subcategory -> mwg.categories.category_id   (62 nhóm, cũng là đơn vị gợi ý)
    group_name  -> mwg.categories.category_name
    category_id -> mwg.categories.parent_category_code (5 mục lớn)
    brand       -> mwg.brand.brand_name
    product     -> mwg.product (FK tới 2 bảng trên)

`room_weights` không được nạp: schema 3 bảng không có cột này. Điểm W(c, r) gán
cứng trong `api/app/services/group_scores.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "dienmayxanh_products.csv"
SCHEMA_PATH = ROOT / "db" / "init" / "01_schema.sql"
SCHEMA = "mwg"
DEFAULT_CONTAINER = "mwgshop-postgres"
DEFAULT_DB = "mwgshop"
DEFAULT_USER = "mwg"

# 5 mục lớn trong CSV — chỉ dùng để báo lỗi khi CSV xuất hiện giá trị lạ.
PARENT_CODES = {"electronics", "appliance", "kitchen_appliance", "household", "personal_care"}

JSON_COLUMNS = ("colors", "tags", "style", "placement", "dimensions")


def clean(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def as_json(text_value, fallback):
    """CSV lưu JSON dạng chuỗi; sai định dạng thì dùng mặc định của cột."""
    raw = clean(text_value)
    if raw is None:
        return json.dumps(fallback, ensure_ascii=False)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return json.dumps(fallback, ensure_ascii=False)
    expected = list if isinstance(fallback, list) else dict
    if not isinstance(parsed, expected):
        return json.dumps(fallback, ensure_ascii=False)
    return json.dumps(parsed, ensure_ascii=False)


def read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_payload(rows: list[dict]) -> dict:
    """Chuẩn hoá CSV -> dữ liệu cho 3 bảng (categories, brand, product)."""
    brands: set[str] = set()
    categories: dict[str, tuple[str, str]] = {}
    products: list[tuple] = []
    skipped: Counter = Counter()

    for index, row in enumerate(rows):
        product_id = clean(row.get("product_id"))
        product_name = clean(row.get("product_name"))
        group_code = clean(row.get("subcategory"))
        group_name = clean(row.get("group_name"))
        parent_code = clean(row.get("category_id"))
        brand = clean(row.get("brand"))
        price_raw = clean(row.get("price"))

        # Mọi cột NOT NULL + FK; thiếu là bản ghi không dùng được.
        missing = [
            name for name, value in (
                ("product_id", product_id), ("product_name", product_name),
                ("subcategory", group_code), ("group_name", group_name),
                ("category_id", parent_code), ("brand", brand), ("price", price_raw),
            ) if value is None
        ]
        if missing:
            skipped[f"thieu_{missing[0]}"] += 1
            continue

        if parent_code not in PARENT_CODES:
            skipped["parent_category_la"] += 1

        brands.add(brand)
        # `subcategory` là khoá của categories: một nhóm chỉ có một group_name
        # và một parent. Lệch nghĩa là CSV không nhất quán -> báo, không tự chọn.
        known = categories.setdefault(group_code, (group_name, parent_code))
        if known != (group_name, parent_code):
            skipped["nhom_khong_nhat_quan"] += 1

        products.append((
            product_id,
            product_name,
            clean(row.get("description")),
            group_code,
            brand,
            int(round(float(price_raw))),
            (clean(row.get("currency")) or "VND").upper()[:3],
            clean(row.get("image_url")),
            clean(row.get("product_url")),
            (clean(row.get("availability")) or "true").lower() == "true",
            as_json(row.get("colors"), []),
            as_json(row.get("tags"), []),
            as_json(row.get("style"), []),
            as_json(row.get("placement"), []),
            as_json(row.get("dimensions"), {}),
        ))

    return {
        "brands": sorted(brands),
        "categories": categories,
        "products": products,
        "skipped": skipped,
    }


def load_into_db(payload: dict, conn) -> dict[str, int]:
    from psycopg2.extras import execute_values

    cur = conn.cursor()

    # 1) Xoá sạch, con trước cha để không vướng FK. DROP cả schema cũ nếu có
    #    (`catalog` là schema của phiên bản trước — giữ lại chỉ gây nhầm lẫn).
    cur.execute("DROP SCHEMA IF EXISTS catalog CASCADE")
    cur.execute(f"DROP TABLE IF EXISTS {SCHEMA}.product CASCADE")
    cur.execute(f"DROP TABLE IF EXISTS {SCHEMA}.brand CASCADE")
    cur.execute(f"DROP TABLE IF EXISTS {SCHEMA}.categories CASCADE")

    # 2) Dựng lại schema từ file init duy nhất.
    cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))

    # 3) Cha trước, con sau. brand_id theo thứ tự chèn -> tra qua RETURNING.
    execute_values(
        cur,
        f"INSERT INTO {SCHEMA}.categories "
        "(category_id, category_name, parent_category_code) VALUES %s",
        [(code, name, parent) for code, (name, parent) in sorted(payload["categories"].items())],
        page_size=1000,
    )
    execute_values(
        cur,
        f"INSERT INTO {SCHEMA}.brand (brand_name) VALUES %s ORDER BY 1",
        [(name,) for name in payload["brands"]],
        page_size=1000,
    )
    cur.execute(f"SELECT brand_name, brand_id FROM {SCHEMA}.brand")
    brand_ids = dict(cur.fetchall())

    rows = [
        (
            product_id, product_name, description, group_code,
            brand_ids[brand], price, currency, image_url, product_url,
            availability, colors, tags, style, placement, dimensions,
        )
        for (
            product_id, product_name, description, group_code, brand, price,
            currency, image_url, product_url, availability, colors, tags,
            style, placement, dimensions,
        ) in payload["products"]
    ]
    execute_values(
        cur,
        f"INSERT INTO {SCHEMA}.product (product_id, product_name, description, "
        "category_id, brand_id, price, currency, image_url, product_url, "
        "availability, colors, tags, style, placement, dimensions) VALUES %s",
        rows,
        page_size=500,
    )

    cur.execute(
        f"ANALYZE {SCHEMA}.product; ANALYZE {SCHEMA}.categories; ANALYZE {SCHEMA}.brand"
    )

    return {
        "categories": len(payload["categories"]),
        "brand": len(payload["brands"]),
        "product": len(rows),
    }


def connect(args):
    import psycopg2

    if args.dsn:
        return psycopg2.connect(args.dsn)

    port = subprocess.run(
        ["docker", "port", args.container, "5432/tcp"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if not port:
        raise SystemExit(
            f"❌ không lấy được cổng của container `{args.container}`.\n"
            f"   Kiểm tra `docker compose ps`, hoặc truyền --dsn."
        )
    return psycopg2.connect(
        host="127.0.0.1", port=port.rsplit(":", 1)[-1],
        dbname=args.db, user=args.user, password=args.password,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", nargs="?", default=str(DEFAULT_CSV))
    parser.add_argument("--dsn")
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=None, help="mặc định đọc POSTGRES_PASSWORD trong .env")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = Path(args.csv).expanduser()
    if not path.is_file():
        print(f"❌ không thấy file: {path}", file=sys.stderr)
        return 1

    print(f"📖 {path}  ({path.stat().st_size / 1_048_576:.1f} MB)")
    rows = read_rows(path)
    if not rows:
        print("❌ CSV rỗng", file=sys.stderr)
        return 1

    payload = build_payload(rows)
    per_category = Counter(p[3] for p in payload["products"])
    print(f"   dòng CSV      : {len(rows):,}")
    print(f"   product       : {len(payload['products']):,}")
    print(f"   categories    : {len(payload['categories']):,}  "
          f"(nhóm nhỏ nhất {min(per_category.values())}, lớn nhất {max(per_category.values())})")
    print(f"   brand         : {len(payload['brands']):,}")
    if payload["skipped"]:
        print("   ⚠️  bỏ qua     : " + ", ".join(f"{k}={v}" for k, v in payload["skipped"].items()))

    if args.dry_run:
        print("✅ dry-run — không ghi DB")
        return 0

    if args.password is None:
        env_path = ROOT / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("POSTGRES_PASSWORD="):
                    args.password = line.split("=", 1)[1].strip()
                    break

    conn = connect(args)
    print(f"🔌 đã kết nối (cổng {conn.get_dsn_parameters().get('port')})")
    try:
        stats = load_into_db(payload, conn)
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        print(
            f"\n❌ lỗi — đã rollback, DB giữ nguyên:\n   {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        conn.close()

    print("\n✅ đã nạp và commit:")
    for key, value in stats.items():
        print(f"   {key:>12}: {value:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
