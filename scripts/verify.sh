#!/usr/bin/env bash
# Kiểm tra toàn vẹn catalog sau khi nạp data/dienmayxanh_products.csv.
#
#   ./scripts/verify.sh
#
# Thoát mã 1 nếu có bất kỳ kiểm tra nào sai.
set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE="docker compose --env-file .env"
PSQL="$COMPOSE exec -T postgres psql -U mwg -d mwgshop -v ON_ERROR_STOP=1 -tA"

run() {
  # In giá trị scalar; thoát nếu psql lỗi.
  $PSQL -c "$1"
}

expect() {
  # expect <mô tả> <sql trả về số> <giá trị mong đợi>
  local label=$1 sql=$2 want=$3 got
  got=$(run "$sql" | tr -d '[:space:]')
  if [[ "$got" == "$want" ]]; then
    printf '  ✅ %-46s %s\n' "$label" "$got"
  else
    printf '  ❌ %-46s %s (mong đợi %s)\n' "$label" "$got" "$want"
    failures=$((failures + 1))
  fi
}

failures=0

echo "📦 catalog — kiểm tra toàn vẹn (schema mwg)"
expect "số sản phẩm" \
  "SELECT count(*) FROM mwg.product" 1413
expect "số category (= 62 nhóm)" \
  "SELECT count(*) FROM mwg.categories" 62
expect "số parent category (5 mục lớn)" \
  "SELECT count(DISTINCT parent_category_code) FROM mwg.categories" 5
expect "số brand" \
  "SELECT count(*) FROM mwg.brand" 157

echo
echo "🔗 khoá ngoại — tất cả phải 0"
expect "sản phẩm mồ côi khỏi category" \
  "SELECT count(*) FROM mwg.product p
   LEFT JOIN mwg.categories c ON c.category_id = p.category_id
   WHERE c.category_id IS NULL" 0
expect "sản phẩm mồ côi khỏi brand" \
  "SELECT count(*) FROM mwg.product p
   LEFT JOIN mwg.brand b ON b.brand_id = p.brand_id
   WHERE b.brand_id IS NULL" 0
expect "brand không được dùng" \
  "SELECT count(*) FROM mwg.brand b
   LEFT JOIN mwg.product p ON p.brand_id = b.brand_id
   WHERE p.product_id IS NULL" 0

echo
echo "🧮 ràng buộc dữ liệu"
expect "giá âm" "SELECT count(*) FROM mwg.product WHERE price < 0" 0
expect "tên sản phẩm rỗng" \
  "SELECT count(*) FROM mwg.product WHERE length(btrim(product_name)) = 0" 0
expect "brand_name rỗng" \
  "SELECT count(*) FROM mwg.brand WHERE length(btrim(brand_name)) = 0" 0
expect "category_name rỗng" \
  "SELECT count(*) FROM mwg.categories WHERE length(btrim(category_name)) = 0" 0
expect "product_id duy nhất" \
  "SELECT count(DISTINCT product_id) FROM mwg.product" 1413
expect "còn hàng (availability)" \
  "SELECT count(*) FROM mwg.product WHERE availability" 1413
expect "product_name trùng description" \
  "SELECT count(*) FROM mwg.product WHERE product_name = description" 1413

echo
echo "🧩 toàn vẹn khoá ngoại theo nhóm"
expect "nhóm có sản phẩm đều có trong mwg.categories" \
  "SELECT count(*) FROM (
     SELECT DISTINCT category_id FROM mwg.product
   ) t WHERE t.category_id NOT IN (SELECT category_id FROM mwg.categories)" 0

echo
echo "🎯 artifact (bắt buộc, không dự phòng)"
for f in api/artifacts/matrix/cooccurrence_matrix.json \
         api/artifacts/matrix/category_room_scores.json; do
  if [[ -f "$f" ]]; then
    printf '  ✅ %-52s %s\n' "$f" "$(du -h "$f" | cut -f1)"
  else
    printf '  ❌ %-52s thiếu\n' "$f"
    failures=$((failures + 1))
  fi
done

echo
if (( failures > 0 )); then
  echo "❌ $failures kiểm tra thất bại"
  exit 1
fi
echo "✅ catalog + artifact toàn vẹn"
