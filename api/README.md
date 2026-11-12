# AI Backend — Gợi ý đồ điện máy & Phối cảnh nội thất

FastAPI backend cho bài toán nhận diện phòng, gợi ý sản phẩm điện máy và phối cảnh vào không gian thực tế, sử dụng **Grounding DINO** (`IDEA-Research/grounding-dino-tiny`, giấy phép **Apache-2.0**), ma trận đồng xuất hiện Noisy-OR 45 class nội thất, bảng điểm phòng $W(c, r)$ 62 nhóm, catalog PostgreSQL 16 và cổng **Qwen-Image-Edit Gateway**.

---

## Kiểm tra Bản quyền & Giấy phép Sử dụng (License & Compliance)

Mọi thành phần trong backend được thiết kế tuân thủ nghiêm ngặt tính hợp pháp và quyền thương mại:

| Thành phần | Tên / Nguồn | Giấy phép | Quyền thương mại | Ghi chú kỹ thuật |
| :--- | :--- | :--- | :---: | :--- |
| **Object Detection** | `IDEA-Research/grounding-dino-tiny` | **Apache-2.0** |  Cho phép | Cho phép nhúng vào hệ thống thương mại kín (proprietary SaaS). Không bị ràng buộc copyleft mạng như Ultralytics (AGPL-3.0). |
| **Frameworks** | Hugging Face `transformers`, `torch` | **Apache-2.0** / **BSD-3** |  Cho phép | Tiêu chuẩn mã nguồn mở thương mại. |
| **Phối cảnh AI** | Qwen-Image-Edit (Tongyi Qianwen) | **Tongyi Qianwen License** / **Apache-2.0** |  Cho phép | Miễn phí cho sản phẩm thương mại dưới 100M MAU. Kết nối qua cổng nội bộ MWG AI Gateway (`ai.itmwg.vn`). |
| **Dữ liệu Ma trận** | MS COCO 2017 Dataset Annotations | **CC BY 4.0** |  Cho phép | Dùng thống kê xác suất đồng xuất hiện từ 1.717 ảnh phòng có ghi nhận nguồn. |
| **Catalog sản phẩm** | MWG Điện Máy Xanh | **Bản quyền MWG** |  Nội bộ | 1.413 sản phẩm thuộc quyền sở hữu của MWG. |
| **API Framework** | FastAPI, Uvicorn, Pydantic, PostgreSQL | **MIT** / **PostgreSQL** |  Cho phép | Các thư viện mã nguồn mở tối ưu cho microservices. |

---

## Kiến trúc Zero-Shot Detection (Không cần Train)

Model phát hiện vật thể không yêu cầu huấn luyện lại hay lưu checkpoint `.pt` độc quyền. Model được tải trực tiếp từ Hugging Face Hub (hoặc snapshot cache cục bộ):

```python
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
```

Bảng ánh xạ `CLASS_PROMPTS` trong `app/services/detector.py` định nghĩa các từ khoá tiếng Anh tự nhiên (synonyms) cho 45 class nội thất chuẩn trong ma trận đồng xuất hiện (`sofa`, `couch`, `bed`, `dining table`, `tv stand`, `air conditioner`...), đảm bảo độ nhạy nhận diện cao và nhất quán.

---

## Bản đồ chuyển giao Notebook → Codebase

| Notebook | Nội dung nghiệp vụ | Module tương ứng |
| :--- | :--- | :--- |
| Cell 1–5 | Đường dẫn, cấu hình | `app/core/config.py` |
| Cell 6 | `CLASS_MERGE_MAP`, `DROP_CLASSES` | `app/services/cooccurrence.py` |
| Cell 7 | Bộ 45 class detector | `expected_detector_classes` trong `app/services/detector.py` |
| **Cell 8** | **Huấn luyện** | **Bỏ (Dùng Zero-Shot Grounding DINO)** |
| Cell 9 | Nạp detector & kiểm tra class | `app/services/detector.py` |
| Cell 10.1 | Hằng số cấu hình | `app/core/config.py` |
| Cell 10.2 | Ánh xạ `GROUP_TO_DETECTOR` | `app/services/recommender.py` |
| Cell 10.3 | `build_cooccurrence_matrix` | `app/services/cooccurrence.py` |
| Cell 10.4 | Ước lượng xác suất $P(e \mid d)$ | `Recommender.p_smoothed` |
| Cell 10.5 | Lưu `cooccurrence_matrix.json` | `artifacts/matrix/cooccurrence_matrix.json` (Artifact tĩnh) |
| Cell 10.6 | Bảng điểm phòng $W(c, r)$ | `app/services/room_scores.py` |
| Cell 10.7 | Hàm tra cứu trọng số $W$ | `Recommender.w` |
| Cell 10.8 | Nạp Catalog | `app/services/product_catalog.py` (Chỉ đọc PostgreSQL) |
| Cell 10.9 | Mô hình Noisy-OR, `build_reason` | `Recommender.cooccurrence_score` |
| Cell 10.10 | Thuật toán `recommend()` | `Recommender.recommend` |
| Cell 10.12 | Đọc và tối ưu ảnh | `app/services/images.py` |
| Cell 10.13 | Nhận diện vật thể | `Detector.detect` |
| Cell 10.14 | Endpoint API | `POST /api/v1/recommendations/image` |
| **Mới** | **Prompt Builder phối cảnh thực tế** | `app/services/prompt_builder.py` |
| **Mới** | **Kết nối Qwen Image Edit Gateway** | `app/services/qwen_image_edit.py` |

---

## Cấu hình Biến Môi trường

| Biến | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `MODEL_PATH` | `IDEA-Research/grounding-dino-tiny` | Hugging Face Repo ID hoặc thư mục cache mô hình |
| `COOCCURRENCE_MATRIX_PATH` | `artifacts/matrix/cooccurrence_matrix.json` | Ma trận đồng xuất hiện 45 class (1.717 ảnh train) |
| `CATEGORY_ROOM_SCORES_PATH` | `artifacts/matrix/category_room_scores.json` | Ma trận điểm phòng 62 nhóm × 3 loại phòng |
| `DB_HOST` / `DB_PORT` | `postgres` / `5432` | Địa chỉ PostgreSQL (host `postgres` trong Docker) |
| `DB_SCHEMA` | `mwg` | Schema chứa 3 bảng (`product`, `categories`, `brand`) |
| `DB_USER` / `DB_PASSWORD` | `mwg` / `mwg_secret` | Thông tin xác thực PostgreSQL |
| `QWEN_GATEWAY_URL` | `https://ai.itmwg.vn/ai-mwg/qwen-edit-image` | URL cổng Qwen Image Edit Gateway |
| `QWEN_API_KEY` | *(Key bảo mật MWG)* | API Key gửi qua header `apikey` |
| `AUTO_GENERATE_IMAGE` | `false` | Tự động tạo ảnh phối cảnh nếu request không chỉ định |

> [!CAUTION]
> **Không có cơ chế fallback giả lập**: Nếu thiếu bất kỳ thành phần nào trong 4 nguồn dữ liệu (Model, Ma trận Cooc, Điểm phòng, PostgreSQL Catalog), API sẽ dừng khởi động ngay lập tức và ghi rõ lỗi trong log.

---

## Dữ liệu: PostgreSQL là Nguồn Duy nhất

Catalog 1.413 sản phẩm được lưu trữ trong schema `mwg` trên PostgreSQL:
- Bảng `categories`: 62 nhóm sản phẩm (`category_id` tương ứng với `subcategory`).
- Bảng `brand`: 157 nhãn hàng.
- Bảng `product`: 1.413 thiết bị kèm liên kết khoá ngoại, giá tiền và link ảnh.

Khởi động hệ thống từ thư mục gốc repo:
```bash
docker compose up -d --build
curl -s localhost:8000/health/ready | python3 -m json.tool
```

Trường `catalog_source` phải trả về `PostgreSQL mwg.product (1413 sản phẩm)`.

---

## Chạy Local (Môi trường Phát triển)

```bash
cd api
# 1. Kích hoạt môi trường ảo Python
source .venv/bin/activate

# 2. Cấu hình biến môi trường kết nối tới DB container
cp .env.example .env
# Chỉnh DB_HOST=127.0.0.1 và DB_PORT=55432 trong file .env

# 3. Chạy server backend
uvicorn app.main:get_app --factory --host 127.0.0.1 --port 8000 --reload
```

---

## API Endpoints

| Phương thức | Đường dẫn | Chức năng |
| :--- | :--- | :--- |
| **POST** | `/api/v1/recommendations/image` | Nhận diện phòng, gợi ý sản phẩm và tuỳ chọn ghép ảnh |
| **GET** | `/health/ready` | Kiểm tra trạng thái sẵn sàng của Model, DB và Ma trận |

### Ví dụ Request:

#### 1. Gợi ý đồ điện máy cho phòng khách:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/recommendations/image?room_type=living_room&top_k=5" \
     -F "image=@phong_khach.jpg"
```

#### 2. Gợi ý kèm phối cảnh sản phẩm vào ảnh phòng:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/recommendations/image?room_type=living_room&generate_image=true" \
     -F "image=@phong_khach.jpg"
```

Các tham số tuỳ chọn:
- `room_type`: 1 trong 3 phòng chuẩn (`living_room`, `bedroom`, `kitchen`).
- `top_k`: Số nhóm sản phẩm gợi ý (mặc định 5).
- `generate_image`: `true` để kích hoạt AI ghép sản phẩm vào ảnh phòng.
- `edit_prompt`: Chuỗi prompt tùy biến nếu muốn ghi đè prompt mặc định.

### Batch ba phòng

`POST /api/v1/recommendations/images/auto` nhận đúng ba field ảnh cùng tên `images`, tự xác định loại phòng từng ảnh, rồi ghép sản phẩm cho từng ảnh:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/recommendations/images/auto?top_k=5&product_per_group=3" \
  -F "images=@phong-1.jpg" -F "images=@phong-2.jpg" -F "images=@phong-3.jpg"
```

`POST /api/v1/recommendations/images/by-room` nhận thêm đúng ba `room_types` theo cùng thứ tự ảnh:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/recommendations/images/by-room?top_k=5&product_per_group=3" \
  -F "images=@phong-khach.jpg" -F "images=@phong-ngu.jpg" -F "images=@bep.jpg" \
  -F "room_types=living_room" -F "room_types=bedroom" -F "room_types=kitchen"
```

---

## Cơ chế Phối cảnh Thông minh (`ImageEditPromptBuilder`)

Hệ thống tích hợp class `ImageEditPromptBuilder` (`app/services/prompt_builder.py`) để kiểm soát chất lượng ảnh sinh ra từ Qwen-Image-Edit qua endpoint `/v1/images/edits`.

### 1. Nguyên tắc cốt lõi:
1. **Bảo tồn cấu trúc phòng (Preserve Room Architecture)**: Giữ nguyên sàn nhà, tường, cửa sổ và mọi đồ nội thất hiện có. Cấm tạo thêm bàn, kệ mới nếu phòng chưa có.
2. **Quy tắc vị trí theo từng nhóm thiết bị (62 nhóm catalog)**:
   - **Tivi (`tv`)**: Nếu sản phẩm có chân đế (`stand`/`legs`), bắt buộc đặt đứng trên kệ tivi hiện có; nếu không có kệ hoặc TV không chân thì mới gắn phẳng trên tường.
   - **Máy điều hòa (`air_conditioner`)**: Bắt buộc gắn trên tường cao sát trần nhà (`mounted high on the upper wall plane near the ceiling`). Tuyệt đối cấm đặt dưới sàn, trên kệ tivi hay mặt bàn.
   - **Thiết bị nhà bếp để bàn (`gas_stove`, `induction_cooker`, `infrared_cooker`, `microwave`, `oven`, `kettle`, `blender`)**: Đặt nằm phẳng ngay ngắn trên mặt bàn bếp hiện có (`existing kitchen counter`) với kích thước thực tế (~20–50cm).
   - **Thiết bị lớn (`refrigerator`, `washing_machine`, `freezer`)**: Đặt thẳng đứng với chân đế đặt vững trên sàn nhà và tựa sát vào tường.
3. **Cấm che khuất tiền cảnh (`Ban giant foreground objects`)**: Cấm đặt các vật thể khổng lồ che lấp góc nhìn phía trước camera hoặc lơ lửng giữa không trung.
4. **Giới hạn số lượng**: Chỉ ghép tối đa 2 sản phẩm từ 2 nhóm gợi ý hàng đầu nhằm tuân thủ giới hạn 3 ảnh (`max_images = 3`) của Gateway.

---

## Công thức Toán học

1. **Xác suất có điều kiện được làm trơn (Smoothed Conditional Probability)**:
   $$\hat{P}(e \mid d) = \frac{N(d \cap e) + \alpha \cdot P(e)}{N(d) + \alpha K}, \quad K=2,\ \alpha=1$$

2. **Mô hình kết hợp Noisy-OR**:
   $$P_{\text{cooc}}(c \mid D) = 1 - \prod_{d \in D}\left(1 - \hat{P}(\pi(c) \mid d)\right)$$

3. **Điểm số cuối cùng (Final Ranking Score)**:
   $$\text{final\_score}(c) = P_{\text{cooc}}(c \mid D) + W(c, r)$$

---

## Kiểm thử (Unit Tests)

Bộ test toàn diện gồm 107 bài test kiểm tra công thức xếp hạng, kết nối DB, cấu hình OpenAPI, bộ quy tắc prompt builder và client Qwen Gateway:

```bash
PYTHONPATH=. pytest tests/ -v
```

Kiểm tra độ dài tài liệu Swagger OpenAPI không vượt giới hạn:
```bash
PYTHONPATH=. pytest tests/test_docs.py
```
