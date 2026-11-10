# Hệ thống Gợi ý & Phối cảnh Đồ điện máy từ Ảnh phòng

Hệ thống kết hợp **Computer Vision (Zero-Shot Object Detection)**, **Mô hình xác suất đồng xuất hiện (Noisy-OR Co-occurrence)**, **Catalog Điện Máy Xanh (PostgreSQL 16)** và **AI Generative Inpainting (Qwen-Image-Edit)** để nhận diện nội thất phòng, gợi ý thiết bị điện máy phù hợp nhất và tự động phối cảnh sản phẩm vào không gian thực tế của người dùng.

```
Ảnh phòng người dùng
        │
        ▼
[Grounding DINO Tiny] ──(Apache-2.0)──> Nhận diện vật thể hiện có (Sofa, TV, Giường, Bàn...)
        │
        ├─ Xác suất đồng xuất hiện P_cooc(c|D) = 1 − Π(1 − P̂(π(c)|d))  (Ma trận COCO 45 class)
        ├─ Điểm tương thích không gian W(c, r)                           (62 nhóm × 7 loại phòng)
        └─ Final Score = P_cooc + W(c, r) ──> Xếp hạng 62 nhóm thiết bị
                                                       │
                                                       ▼
[PostgreSQL 16 Catalog] ───────────────> Lọc top sản phẩm theo giá & thương hiệu
                                                       │
                                                       ▼
[ImageEditPromptBuilder] ──────────────> Sinh prompt chuẩn hoá vị trí & tỉ lệ thực tế
                                                       │
                                                       ▼
[Qwen-Image-Edit Gateway] ─────────────> Ảnh phòng thực tế đã được phối cảnh sản phẩm
```

---

## Bảng kiểm tra Bản quyền & Giấy phép Sử dụng (License Audit)

Toàn bộ các mô hình, thư viện và dữ liệu trong dự án được rà soát để đảm bảo **an toàn tuyệt đối cho mục đích thương mại (Commercial Friendly)** và tuân thủ chính sách bản quyền doanh nghiệp:

| Thành phần | Công nghệ / Nguồn | Giấy phép (License) | Dùng thương mại | Đánh giá & Rủi ro pháp lý |
| :--- | :--- | :--- | :---: | :--- |
| **Object Detector** | `IDEA-Research/grounding-dino-tiny` | **Apache-2.0** |  Cho phép | **An toàn tuyệt đối.** Không có điều khoản copyleft (khác với Ultralytics YOLO/RT-DETR dùng AGPL-3.0 đòi hỏi phải mở toàn bộ mã nguồn backend). Cho phép đóng gói SaaS thương mại. |
| **AI Inference** | Hugging Face `transformers`, `torch` | **Apache-2.0** / **Modified BSD** |  Cho phép | Các framework mã nguồn mở tiêu chuẩn công nghiệp cho production. |
| **Generative Staging** | Qwen-Image-Edit (Alibaba Cloud / Tongyi) | **Tongyi Qianwen License** / **Apache-2.0** |  Cho phép | Cho phép ứng dụng thương mại miễn phí (< 100M MAU). Triển khai qua cổng nội bộ MWG AI Gateway (`ai.itmwg.vn`). |
| **Ma trận thống kê** | MS COCO Dataset Annotations (2017) | **CC BY 4.0** |  Cho phép | Cho phép sử dụng và trích xuất ma trận tần suất đồng xuất hiện cho mục đích thương mại với điều kiện ghi nhận tác giả. |
| **Catalog sản phẩm** | Dien May Xanh (`dienmayxanh_products.csv`) | **Bản quyền MWG** |  Nội bộ | 1.413 sản phẩm, 62 nhóm ngành hàng thuộc sở hữu của Tập đoàn Thế Giới Di Động. |
| **Database & API** | FastAPI, PostgreSQL 16, Pydantic | **MIT** / **PostgreSQL License** |  Cho phép | Toàn bộ stack phần mềm nền tảng sử dụng các giấy phép mã nguồn mở tự do nhất. |

> [!IMPORTANT]
> **Loại bỏ rủi ro AGPL-3.0**: Trước đây, các kiến trúc dựa trên `ultralytics` (YOLOv8, RT-DETR) mang giấy phép **AGPL-3.0** (buộc phải phát hành toàn bộ mã nguồn khi cung cấp dịch vụ qua mạng). Hệ thống hiện tại đã chuyển dịch hoàn toàn sang **Grounding DINO Tiny (Apache-2.0)** qua Hugging Face Hub, đảm bảo 100% mã nguồn và API của doanh nghiệp được giữ kín (Proprietary).

---

## Cổng dịch vụ

| Dịch vụ | URL | Thông tin xác thực / Ghi chú |
| :--- | :--- | :--- |
| **API Backend** | http://localhost:8000 | Swagger UI: `/docs` · ReDoc: `/redoc` |
| **PostgreSQL** | `localhost:55432` | User: `mwg` / Pass: `mwg_secret` / DB: `mwgshop` |
| **Adminer UI** | http://localhost:58080 | Hệ quản trị DB qua web (Server: `postgres`) |

---

## Khởi động nhanh (Quick Start)

### 1. Khởi động toàn bộ hệ thống bằng Docker

```bash
docker compose up -d --build     # Khởi động PostgreSQL, FastAPI Backend và Adminer
```

Kiểm tra trạng thái sẵn sàng của 4 thành phần bắt buộc qua endpoint `/health/ready`:

```bash
curl -s localhost:8000/health/ready | python3 -m json.tool
```

Phản hồi mẫu xác nhận sẵn sàng:
```json
{
  "status": "ready",
  "model_source": "IDEA-Research/grounding-dino-tiny (grounding_dino)",
  "cooccurrence_source": "cooccurrence_matrix.json (45 class, 1717 ảnh train)",
  "room_scores_source": "artifact category_room_scores.json (62 nhóm × 7 phòng) + mwg.categories",
  "catalog_source": "PostgreSQL mwg.product (1413 sản phẩm)"
}
```

*Nguyên tắc nghiêm ngặt: Nếu thiếu 1 trong 4 nguồn trên, hệ thống sẽ từ chối khởi động và ghi log chi tiết lỗi, không sử dụng dữ liệu giả lập.*

---

### 2. Gợi ý sản phẩm và Tự động ghép ảnh phòng

#### Trường hợp 1: Chỉ lấy danh sách gợi ý sản phẩm
```bash
curl -X POST "http://localhost:8000/api/v1/recommendations/image?room_type=living_room&top_k=3" \
     -F "image=@phong-khach.jpg"
```

#### Trường hợp 2: Gợi ý kèm Tự động phối cảnh sản phẩm vào phòng (`generate_image=true`)
```bash
curl -X POST "http://localhost:8000/api/v1/recommendations/image?room_type=living_room&top_k=2&generate_image=true" \
     -F "image=@phong-khach.jpg"
```

Phản hồi trả về bao gồm:
- Danh sách thiết bị điện máy phù hợp nhất (Tên sản phẩm, giá, hình ảnh, lý do gợi ý).
- Trường `generated_image`: Ảnh phòng thực tế đã được ghép thiết bị với tỉ lệ và vị trí tự nhiên.
- Trường `generation_status`: `"success"` hoặc `"failed"`.

---

## Cấu trúc Dự án

```
recommend-by-image-v7/
├── docker-compose.yml            # Điều phối postgres + api + adminer
├── Makefile                      # Các lệnh make thao tác nhanh
├── db/
│   └── init/
│       ├── 01_schema.sql         # Schema `mwg`: categories, brand, product (3 bảng chuẩn)
│       └── 02_seed.sql           # Nạp 1.413 sản phẩm catalog MWG
├── data/
│   └── dienmayxanh_products.csv  # Nguồn catalog: 1.413 sản phẩm, 62 nhóm, 157 nhãn hàng
├── scripts/
│   ├── import_catalog.py         # Nạp catalog CSV vào PostgreSQL trong 1 transaction
│   ├── gen_seed_sql.py           # Sinh file seed SQL từ CSV
│   ├── gen_group_scores.py       # Khởi tạo ma trận điểm phòng W(c, r) từ CSV
│   └── verify.sh                 # Kiểm tra tính toàn vẹn của dữ liệu và artifact
├── api/                          # FastAPI Backend
│   ├── Dockerfile                # Image dựa trên python:3.14-slim, PyTorch CPU
│   ├── artifacts/matrix/         # Ma trận đồng xuất hiện và bảng điểm phòng
│   └── app/
│       ├── api/routes/           # Endpoint gợi ý và healthcheck
│       ├── core/                 # Cấu hình môi trường (Pydantic Settings)
│       ├── schemas/              # Khai báo schema Pydantic request/response
│       └── services/
│           ├── detector.py       # Zero-shot Object Detector Grounding DINO
│           ├── recommender.py    # Thuật toán tính điểm Noisy-OR và xếp hạng
│           ├── prompt_builder.py # Engine tạo prompt ghép ảnh thông minh (62 nhóm quy tắc)
│           └── qwen_image_edit.py# Client tích hợp cổng AI Gateway Qwen-Image-Edit
```

---

## Cơ sở Dữ liệu & Quy tắc Xếp hạng

### 1. Schema `mwg` (PostgreSQL 16)
Dữ liệu lưu trữ trong 3 bảng có liên kết khoá ngoại chặt chẽ:
- **`categories`**: 62 subcategory (`tv`, `air_conditioner`, `refrigerator`...) thuộc 5 ngành hàng lớn (`parent_category_code`).
- **`brand`**: 157 thương hiệu chuẩn hoá.
- **`product`**: 1.413 sản phẩm kèm giá, link ảnh (`image_url`), thông số JSONB.

### 2. Thuật toán Xếp hạng Gợi ý
Điểm số cuối cùng của một nhóm sản phẩm $c$ trong phòng có các vật thể phát hiện $D$ được tính bằng:
$$\text{final\_score}(c) = P_{\text{cooc}}(c \mid D) + W(c, r)$$

Trong đó:
1. **$P_{\text{cooc}}(c \mid D) = 1 - \prod_{d \in D}\left(1 - \hat{P}(\pi(c) \mid d)\right)$**: Xác suất đồng xuất hiện tính theo mô hình Noisy-OR, dựa trên tần suất thực tế từ tập dữ liệu phòng khách/phòng ngủ COCO. Nhóm đồ điện máy nào đã xuất hiện trong ảnh sẽ bị loại trừ.
2. **$W(c, r) \in [0, 1]$**: Trọng số mức độ phù hợp giữa nhóm sản phẩm $c$ và công năng của loại phòng $r$ (phòng khách, nhà bếp, phòng ngủ, phòng làm việc...).

### 3. Công nghệ Phối cảnh Thông minh (`ImageEditPromptBuilder`)
Để giải quyết bài toán AI tạo hình làm biến dạng phòng hoặc đặt vật thể sai vị trí:
- **Nguyên tắc bảo toàn cấu trúc phòng**: Cấm AI tự ý vẽ thêm bàn, tủ, kệ mới nếu phòng chưa có.
- **Quy tắc vị trí chính xác cho 62 nhóm đồ**:
  - *Tivi*: Tự động phát hiện nếu có chân đế thì đặt đứng trên kệ tivi hiện hữu; nếu không có kệ thì treo tường phẳng.
  - *Máy lạnh*: Bắt buộc gắn trên tường cao sát trần nhà; tuyệt đối không đặt dưới sàn hay trên kệ.
  - *Thiết bị bếp để bàn (nồi cơm, nồi chiên, ấm siêu tốc, bếp từ)*: Đặt ngay ngắn trên mặt bàn bếp hiện có với kích thước thực tế (~30–40cm).
  - *Thiết bị lớn (tủ lạnh, máy giặt)*: Đứng vững trên mặt sàn, tựa sát vào tường.
- **Chống che khuất cận cảnh**: Cấm AI đặt đồ đạc khổng lồ ở tiền cảnh sát ống kính camera (`ban giant foreground objects`).

---

## Các lệnh Quản trị Thường dùng

```bash
make up        # Khởi động toàn bộ container và kiểm tra /health/ready
make logs      # Theo dõi log thời gian thực của FastAPI API
make import    # Xoá trắng DB và nạp lại 1.413 sản phẩm từ CSV gốc
make scores    # Cập nhật lại ma trận điểm phòng khi có thay đổi nghiệp vụ
make verify    # Chạy script tự động kiểm tra toàn vẹn DB và artifact
make psql      # Mở terminal psql kết nối trực tiếp vào Postgres container
make down      # Dừng toàn bộ dịch vụ (giữ nguyên dữ liệu volume)
make reset     # Dừng, xoá volume DB và khởi tạo lại từ đầu
```
