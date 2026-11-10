#!/usr/bin/env python3
"""Sinh `db/init/02_seed.sql` từ `data/dienmayxanh_products.csv`.

Dùng cho docker: `postgres` chạy các file trong `db/init` một lần khi volume còn
trống, và file seed nằm ở đó nên container tự có 1.413 sản phẩm mà không cần
thêm bước thủ công.

Vì sao không mount `scripts/import_catalog.py` vào entrypoint: entrypoint chỉ
chạy `*.sql` bằng `psql`, image `postgres:16-alpine` không có Python. Sinh SQL
sẵn là cách duy nhất để `docker compose up` xong là DB có dữ liệu.

Ánh xạ CSV -> schema 3 bảng (`mwg`):

    CSV `subcategory` (62)   -> mwg.categories.category_id     (khoá chính)
    CSV `group_name` (62)    -> mwg.categories.category_name
    CSV `category_id` (5)    -> mwg.categories.parent_category_code
    CSV `brand` (157)        -> mwg.brand.brand_name
    CSV `subcategory`        -> mwg.product.category_id (FK)
    CSV `brand`              -> mwg.product.brand_id    (FK)

`room_weights` KHÔNG vào DB: schema 3 bảng không có cột này, và API xếp hạng
bằng bảng gán cứng `api/app/services/group_scores.py` (sinh từ chính cột đó
bằng `scripts/gen_group_scores.py`).

Chạy lại mỗi khi CSV đổi:

    python3 scripts/gen_seed_sql.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "dienmayxanh_products.csv"
DEFAULT_OUT = ROOT / "db" / "init" / "02_seed.sql"

SCHEMA = "mwg"
ROWS_PER_INSERT = 200


def lit(value) -> str:
    """Giá trị SQL: NULL, bool, số hoặc chuỗi escape bằng `''`."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def jsonb(raw, fallback) -> str:
    """Cột JSONB: đọc lại rồi ghi chuẩn hoá, cast tường minh để PG không đoán."""
    if raw is None or not str(raw).strip():
        value = fallback
    else:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = fallback
        expected = list if isinstance(fallback, list) else dict
        value = parsed if isinstance(parsed, expected) else fallback
    return lit(json.dumps(value, ensure_ascii=False)) + "::jsonb"


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def build(csv_path: Path) -> str:
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"CSV rỗng: {csv_path}")

    categories: dict[str, tuple[str, str]] = {}
    brands: dict[str, int] = {}
    products: list[tuple] = []

    for row in rows:
        group_code = row["subcategory"].strip()
        group_name = row["group_name"].strip()
        parent_code = row["category_id"].strip()
        brand = row["brand"].strip()

        categories.setdefault(group_code, (group_name, parent_code))
        brands.setdefault(brand, 0)

        products.append((
            lit(row["product_id"].strip()),
            lit(row["product_name"].strip()),
            lit(row["description"].strip() or None),
            lit(group_code),
            0,  # placeholder, gán brand_id sau khi biết danh sách brand
            lit(int(round(float(row["price"])))),
            lit((row["currency"].strip() or "VND").upper()[:3]),
            lit(row["image_url"].strip() or None),
            lit(row["product_url"].strip() or None),
            lit(row["availability"].strip().lower() == "true"),
            jsonb(row["colors"], []),
            jsonb(row["tags"], []),
            jsonb(row["style"], []),
            jsonb(row["placement"], []),
            jsonb(row["dimensions"], {}),
        ))

    # brand_id tường minh 1..N theo thứ tự bảng chữ cái -> seed chạy lại cho
    # cùng kết quả, dễ so sánh khi review diff.
    for index, name in enumerate(sorted(brands), start=1):
        brands[name] = index
    products = [
        row[:4] + (lit(brands[rows[i]["brand"].strip()]),) + row[5:]
        for i, row in enumerate(products)
    ]

    total = len(rows)
    out: list[str] = [
        "-- Sinh tự động bởi scripts/gen_seed_sql.py — ĐỪNG SỬA TAY.",
        f"-- Nguồn: {csv_path.name}",
        f"-- {total} sản phẩm / {len(categories)} category (= nhóm) / "
        f"{len({c[1] for c in categories.values()})} parent / {len(brands)} brand.",
        "--",
        "-- File này nằm trong db/init nên postgres chạy nó một lần khi volume còn",
        "-- trống; sửa CSV thì chạy lại `python3 scripts/gen_seed_sql.py` rồi xoá",
        "-- volume để nạp lại. Idempotent: TRUNCATE trước khi insert.",
        "",
        "BEGIN;",
        "",
        f"TRUNCATE {SCHEMA}.product, {SCHEMA}.brand, {SCHEMA}.categories "
        "RESTART IDENTITY CASCADE;",
        "",
    ]

    out.append("-- categories: 62 nhóm, category_name = group_name, parent = 5 mục lớn")
    for batch in chunks(sorted(categories.items()), ROWS_PER_INSERT):
        values = ",\n".join(
            f"({lit(code)}, {lit(name)}, {lit(parent)})" for code, (name, parent) in batch
        )
        out.append(
            f"INSERT INTO {SCHEMA}.categories "
            "(category_id, category_name, parent_category_code) VALUES\n"
            f"{values};"
        )
    out.append("")

    out.append("-- brand: 157 hãng, brand_id tường minh 1..N")
    for batch in chunks(sorted(brands.items(), key=lambda kv: kv[1]), ROWS_PER_INSERT):
        values = ",\n".join(f"({lit(i)}, {lit(name)})" for name, i in batch)
        out.append(
            f"INSERT INTO {SCHEMA}.brand (brand_id, brand_name) VALUES\n{values};"
        )
    out.append(
        f"SELECT setval(pg_get_serial_sequence('{SCHEMA}.brand', 'brand_id'), "
        f"(SELECT max(brand_id) FROM {SCHEMA}.brand));"
    )
    out.append("")

    out.append("-- product")
    columns = (
        "product_id, product_name, description, category_id, brand_id, price, "
        "currency, image_url, product_url, availability, colors, tags, style, "
        "placement, dimensions"
    )
    for batch in chunks(products, ROWS_PER_INSERT):
        values = ",\n".join("(" + ", ".join(row) + ")" for row in batch)
        out.append(f"INSERT INTO {SCHEMA}.product ({columns}) VALUES\n{values};")
    out.append("")

    out.append("COMMIT;")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.is_file():
        raise SystemExit(f"❌ không thấy CSV: {csv_path}")

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sql = build(csv_path)
    out_path.write_text(sql, encoding="utf-8")
    print(f"✅ {out_path}  ({len(sql) / 1024:.0f} KB, {sql.count(chr(10))} dòng)")

    # Docker build context là `./api`, nên image cần một bản CSV trong đó để API
    # fallback khi không có DB. Root `data/` là bản gốc, `api/data/` là bản sao.
    image_copy = ROOT / "api" / "data" / csv_path.name
    image_copy.parent.mkdir(parents=True, exist_ok=True)
    if image_copy.resolve() != csv_path.resolve():
        if not image_copy.is_file() or image_copy.read_bytes() != csv_path.read_bytes():
            image_copy.write_bytes(csv_path.read_bytes())
            print(f"🔄 đồng bộ {image_copy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
