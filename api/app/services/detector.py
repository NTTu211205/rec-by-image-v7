"""Tầng 0 — detector Grounding DINO (Zero-Shot Object Detection, Apache-2.0).

Thay thế hoàn toàn checkpoint Ultralytics RT-DETR/YOLO (AGPL-3.0) bằng
`IDEA-Research/grounding-dino-tiny` chạy qua Hugging Face `transformers`
(giấy phép Apache-2.0, sử dụng thương mại tự do).

Grounding DINO nhận danh sách nhãn qua text prompt tiếng Anh, phát hiện vật thể
zero-shot và ánh xạ ngược về đúng 45 class chuẩn của `cooccurrence_matrix.json`.
"""

from __future__ import annotations

import inspect
import logging
import re
import threading
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.core.exceptions import ArtifactError, ModelNotReadyError

logger = logging.getLogger("app.detector")

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_TEXT_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Ánh xạ class chuẩn của detector -> cụm từ prompt tiếng Anh cho Grounding DINO.
# Bao gồm:
# 1) 45 class nội thất chuẩn (khớp 1-1 ma trận đồng xuất hiện `cooccurrence_matrix.json`).
# 2) Các nhóm danh mục chuyên dụng và tiện ích cho Nhà bếp (Kitchen).
# Các class gộp hoặc chứa dấu `/` (như `Basket/Bin`, `Bench/Ottoman`, `Mugs Teacups`...)
# được tách thành các danh từ tự nhiên để BERT text encoder của Grounding DINO bắt chính xác,
# sau đó kết quả được quy đổi ngược về đúng tên chuẩn.
# ---------------------------------------------------------------------------
CLASS_PROMPTS: dict[str, tuple[str, ...]] = {
    # --- 45 class nội thất cơ bản (COCO indoor / Matrix) ---
    "Accent Chair": ("accent chair", "armchair"),
    "Air Conditioner": ("air conditioner",),
    "Basket/Bin": ("basket", "storage bin", "trash bin", "trash can", "garbage can"),
    "Bed": ("bed", "bed frame", "headboard"),
    "Bench/Ottoman": ("bench", "ottoman"),
    "Books": ("books",),
    "Cabinet": ("cabinet", "dresser", "kitchen cabinet", "cupboard", "kitchen counter"),
    "Candle": ("candle",),
    "Chair": ("chair", "dining chair"),
    "Clock": ("clock", "wall clock"),
    "Coffee Table": ("coffee table",),
    "Curtain": ("curtain",),
    "Curtain Rod": ("curtain rod",),
    "Decor": ("statue", "globe", "decor"),
    "Desk Lamp": ("desk lamp",),
    "Entertainment Center": ("tv stand", "entertainment center"),
    "Fan": ("fan", "ceiling fan"),
    "Fireplace": ("fireplace",),
    "Floor Lamp": ("floor lamp",),
    "Laptop": ("laptop",),
    "Light": ("ceiling light", "light fixture"),
    "Mirror": ("mirror",),
    "Monitor": ("monitor",),
    "Mugs Teacups": ("mug", "tea cup"),
    "Organizer Storage": ("storage box", "organizer"),
    "Phone": ("phone", "smartphone"),
    "Picture Frame": ("picture frame",),
    "Pillow": ("pillow", "throw pillow"),
    "Plant": ("potted plant", "plant"),
    "Planter": ("flower pot", "planter"),
    "Power Surge Protector": ("power strip", "surge protector"),
    "Printer": ("printer",),
    "Remote": ("remote control", "remote"),
    "Rug": ("rug", "carpet"),
    "Shelf": ("shelf", "bookshelf"),
    "Side Table": ("side table", "nightstand"),
    "Sofa": ("sofa", "couch"),
    "Speaker": ("speaker", "soundbar"),
    "Stool": ("stool", "bar stool"),
    "TV": ("tv", "television"),
    "Table": ("dining table", "table"),
    "Table Lamp": ("table lamp",),
    "Tableware": ("bowl", "serving tray", "tableware"),
    "Throw Blanket": ("throw blanket", "blanket"),
    "Vase": ("vase",),
    "Wall Art": ("wall art", "painting"),

    # --- Danh mục thiết bị & gia dụng Nhà bếp (Kitchen Categories) ---
    "Refrigerator": ("refrigerator", "fridge"),
    "Freezer": ("freezer", "chest freezer"),
    "Beverage Cooler": ("beverage cooler", "drink cooler", "wine cooler"),
    "Gas Stove": ("gas stove", "gas cooker", "gas range"),
    "Induction Cooker": ("induction cooker", "induction stove", "cooktop"),
    "Infrared Cooker": ("infrared cooker", "infrared stove"),
    "Microwave": ("microwave", "microwave oven"),
    "Oven": ("oven", "electric oven"),
    "Range Hood": ("range hood", "kitchen hood", "exhaust hood"),
    "Dishwasher": ("dishwasher",),
    "Rice Cooker": ("rice cooker",),
    "Air Fryer": ("air fryer",),
    "Kettle": ("electric kettle", "kettle", "water kettle"),
    "Blender": ("blender", "smoothie maker"),
    "Juicer": ("juicer", "fruit juicer"),
    "Soy Milk Maker": ("soy milk maker", "nut milk maker"),
    "Coffee Maker": ("coffee maker", "coffee machine", "espresso machine"),
    "Water Purifier": ("water purifier", "water filter"),
    "Water Dispenser": ("water dispenser", "water cooler"),
}

# ---------------------------------------------------------------------------
# Danh sách class theo từng loại phòng chuyên biệt.
# Giúp Grounding DINO chỉ tìm kiếm các vật thể có liên quan đến phòng đó:
# 1) Giảm thiểu false positive (vd: không bắt nhầm máy hút mùi/nồi cơm trong phòng khách).
# 2) Rút ngắn chuỗi prompt (< 256 tokens), tăng tốc độ suy luận.
# ---------------------------------------------------------------------------
LIVING_ROOM_CLASSES: tuple[str, ...] = (
    # Nội thất & đồ rời phòng khách (COCO indoor)
    "Accent Chair",
    "Air Conditioner",
    "Basket/Bin",
    "Bench/Ottoman",
    "Books",
    "Cabinet",
    "Candle",
    "Chair",
    "Clock",
    "Coffee Table",
    "Curtain",
    "Curtain Rod",
    "Decor",
    "Desk Lamp",
    "Entertainment Center",
    "Fan",
    "Fireplace",
    "Floor Lamp",
    "Laptop",
    "Light",
    "Mirror",
    "Monitor",
    "Mugs Teacups",
    "Organizer Storage",
    "Phone",
    "Picture Frame",
    "Pillow",
    "Plant",
    "Planter",
    "Power Surge Protector",
    "Printer",
    "Remote",
    "Rug",
    "Shelf",
    "Side Table",
    "Sofa",
    "Speaker",
    "Stool",
    "TV",
    "Table",
    "Table Lamp",
    "Tableware",
    "Throw Blanket",
    "Vase",
    "Wall Art",
    # Tiện ích gia dụng có thể đặt ở phòng khách
    "Beverage Cooler",
    "Coffee Maker",
    "Water Dispenser",
)

KITCHEN_CLASSES: tuple[str, ...] = (
    # 19 Thiết bị & gia dụng nhà bếp
    "Refrigerator",
    "Freezer",
    "Beverage Cooler",
    "Gas Stove",
    "Induction Cooker",
    "Infrared Cooker",
    "Microwave",
    "Oven",
    "Range Hood",
    "Dishwasher",
    "Rice Cooker",
    "Air Fryer",
    "Kettle",
    "Blender",
    "Juicer",
    "Soy Milk Maker",
    "Coffee Maker",
    "Water Purifier",
    "Water Dispenser",
    # Đồ dùng, nội thất & tiện ích phòng bếp / ăn
    # "Cabinet",
    # "Table",
    # "Chair",
    # "Stool",
    # "Tableware",
    # "Mugs Teacups",
    # "Shelf",
    # "Basket/Bin",
    # "Organizer Storage",
    # "Light",
    # "Fan",
    # "Air Conditioner",
    # "Clock",
    # "Power Surge Protector",
    # "Plant",
    # "Planter",
    # "Curtain",
    # "Wall Art",
)

BEDROOM_CLASSES: tuple[str, ...] = (
    # Vật thể neo bố cục phòng ngủ.
    "Bed",
    "Pillow",
    "Throw Blanket",
    "Side Table",
    "Table Lamp",
    "Desk Lamp",
    "Floor Lamp",
    "Mirror",
    "Cabinet",
    "Shelf",
    "Rug",
    "Curtain",
    "Curtain Rod",
    "Chair",
    "Table",
    "Books",
    "Picture Frame",
    "Wall Art",
    "Decor",
    "Plant",
    "Planter",
    "Organizer Storage",
    "Laptop",
    "Monitor",
    "Air Conditioner",
    "Fan",
    "Light",
)

ROOM_CLASSES: dict[str, tuple[str, ...]] = {
    "living_room": LIVING_ROOM_CLASSES,
    "kitchen": KITCHEN_CLASSES,
    "bedroom": BEDROOM_CLASSES,
}

def _clean_text(text: str) -> str:
    """Chuẩn hoá chuỗi nhãn: chữ thường, bỏ ký tự đặc biệt thừa, gộp khoảng trắng."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(cleaned.split())


def _build_lookup() -> tuple[dict[str, str], list[tuple[str, str]]]:
    exact: dict[str, str] = {}
    for canonical, phrases in CLASS_PROMPTS.items():
        exact[canonical.lower()] = canonical
        exact[_clean_text(canonical)] = canonical
        for phrase in phrases:
            exact[phrase.lower()] = canonical
            exact[_clean_text(phrase)] = canonical

    # Sắp xếp cụm từ dài trước để khi Grounding DINO ghép nhiều token
    # (vd: "coffee table", "accent chair", "floor lamp", "curtain rod")
    # luôn ưu tiên khớp cụm từ dài cụ thể trước danh từ đơn ("table", "chair", "curtain").
    ordered_phrases = sorted(
        ((k, v) for k, v in exact.items() if k),
        key=lambda item: (-len(item[0].split()), -len(item[0]), item[0]),
    )
    return exact, ordered_phrases


PROMPT_TO_CANONICAL, _ORDERED_PHRASES = _build_lookup()


def build_prompt(classes: list[str] | tuple[str, ...] | None = None) -> str:
    """Tạo chuỗi prompt chuẩn Grounding DINO (cách nhau bởi ` . `, kết thúc bằng `.`)."""
    target_classes = classes if classes is not None else tuple(CLASS_PROMPTS.keys())
    phrases: list[str] = []
    seen: set[str] = set()
    for cls_name in target_classes:
        for phrase in CLASS_PROMPTS.get(cls_name, (cls_name.lower(),)):
            norm = phrase.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                phrases.append(norm)
    return " . ".join(phrases) + " ."


def build_chunked_prompts(
    classes: list[str] | tuple[str, ...] | None = None,
    chunk_size: int = 32,
) -> list[str]:
    """Tách danh sách class thành các chuỗi prompt < 180 tokens để an toàn với giới hạn 256 của BERT."""
    target_classes = list(classes if classes is not None else CLASS_PROMPTS.keys())
    chunks: list[str] = []
    for i in range(0, len(target_classes), chunk_size):
        chunk_classes = target_classes[i : i + chunk_size]
        p = build_prompt(chunk_classes)
        if p.strip() and p.strip() != ".":
            chunks.append(p)
    return chunks


def normalize_label(raw_label: str) -> str | None:
    """Quy đổi nhãn thô do Grounding DINO giải mã về đúng tên trong 45 class chuẩn."""
    if not raw_label:
        return None
    raw_lower = raw_label.strip().lower()
    if raw_lower in PROMPT_TO_CANONICAL:
        return PROMPT_TO_CANONICAL[raw_lower]

    cleaned = _clean_text(raw_lower)
    if not cleaned:
        return None
    if cleaned in PROMPT_TO_CANONICAL:
        return PROMPT_TO_CANONICAL[cleaned]

    padded = f" {cleaned} "
    for phrase, canonical in _ORDERED_PHRASES:
        if f" {phrase} " in padded:
            return canonical
    return None


def resolve_device(requested: str) -> str:
    """auto -> cuda > mps > cpu. Giá trị số ('0') đổi sang 'cuda:0' cho PyTorch."""
    req = (requested or "auto").strip()
    if req != "auto":
        return f"cuda:{req}" if req.isdigit() else req
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class Detector:
    """Bọc Grounding DINO, hỗ trợ detect một ảnh hoặc batch ảnh cùng text prompt."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_ID,
        device: str = "auto",
        jpeg_roundtrip: bool = True,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        batch_size: int = 0,
        shortest_edge: int = 512,
        longest_edge: int = 768,
    ):
        self.model_path = Path(model_path) if isinstance(model_path, Path) else Path(str(model_path))
        self.model_id = str(model_path).strip() or DEFAULT_MODEL_ID
        self.device = resolve_device(device)
        self.jpeg_roundtrip = jpeg_roundtrip
        self.text_threshold = text_threshold
        self.batch_size = batch_size
        self.image_size = {
            "shortest_edge": shortest_edge,
            "longest_edge": longest_edge,
        }
        self.family: str | None = None
        self._processor = None
        self._model = None
        self._classes: list[str] = list(CLASS_PROMPTS.keys())
        self.room_prompts: dict[str, list[str]] = {
            room: build_chunked_prompts(classes, chunk_size=25)
            for room, classes in ROOM_CLASSES.items()
        }
        self.room_prompts["default"] = build_chunked_prompts(self._classes, chunk_size=25)
        self.prompts: list[str] = self.room_prompts.get("living_room") or self.room_prompts["default"]
        self.prompt: str = self.prompts[0] if self.prompts else ""
        self._post_process_uses_threshold: bool = False
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def class_names(self) -> list[str]:
        return list(self._classes)

    def load(self) -> None:
        target = self.model_id
        # Nếu cấu hình trỏ tới đường dẫn cục bộ dạng file `.pt` cũ hoặc thư mục không tồn tại:
        if target.endswith(".pt"):
            raise ArtifactError(
                f"Không hỗ trợ checkpoint Ultralytics ({Path(target).name}); "
                f"hãy dùng model Grounding DINO (vd: '{DEFAULT_MODEL_ID}')."
            )

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(target)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(target)
            self._model.to(self.device)
            self._model.eval()
            self.family = "grounding_dino"

            sig = inspect.signature(self._processor.post_process_grounded_object_detection)
            self._post_process_uses_threshold = "threshold" in sig.parameters
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(
                f"Không nạp được model Grounding DINO ({target}): {type(exc).__name__}: {exc}"
            ) from exc

        logger.info(
            "Đã nạp model Grounding DINO",
            extra={
                "event": "model_loaded",
                "family": self.family,
                "model_id": self.model_id,
                "device": self.device,
                "classes": len(self._classes),
                "detect_image_size": self.image_size,
            },
        )

    def check_classes(self, expected: list[str]) -> None:
        """Bộ class của detector phải bao phủ đầy đủ ma trận đồng xuất hiện."""
        actual = set(self.class_names)
        missing = sorted(set(expected) - actual)
        if missing:
            raise ArtifactError(
                "Class của detector thiếu so với ma trận đồng xuất hiện: "
                f"(thiếu {missing[:5]})."
            )

    def warmup(self) -> None:
        if self.loaded:
            self.detect(Image.new("RGB", (64, 64)), confidence=0.99, room_type="living_room")

    def _prepare_image(self, img: Image.Image) -> Image.Image:
        """Chuẩn hoá đầu vào giống nhau cho đường một ảnh và batch ảnh."""
        if self.jpeg_roundtrip:
            buf = BytesIO()
            img.save(buf, format="JPEG")
            return Image.open(BytesIO(buf.getvalue())).convert("RGB")
        return img if img.mode == "RGB" else img.convert("RGB")

    def _resolve_prompts(self, room_type: str | None) -> tuple[str, list[str]]:
        """Trả room đã chuẩn hoá và các prompt chunks dùng cho room đó."""
        resolved_room = room_type.strip().lower() if room_type else "default"
        prompts = (
            self.room_prompts.get(resolved_room)
            or self.room_prompts.get("default")
            or self.prompts
        )
        return resolved_room, prompts

    @property
    def inference_batch_size(self) -> int:
        """Batch size thực tế: CUDA dùng batch 3, CPU/MPS mặc định dùng từng ảnh.

        Grounding DINO phải padding toàn bộ ảnh trong micro-batch về kích thước
        lớn nhất. Trên CPU, batch 3 thường chậm hơn nhiều so với ba forward nhỏ
        dù số lần gọi model ít hơn; chỉ CUDA mặc định hưởng lợi từ vectorisation.
        ``DETECT_BATCH_SIZE`` cho phép ghi đè sau khi benchmark hạ tầng thực tế.
        """
        if self.batch_size > 0:
            return self.batch_size
        return 3 if self.device.startswith("cuda") else 1

    def detect(
        self,
        img: Image.Image,
        confidence: float,
        room_type: str | None = None,
    ) -> list[dict]:
        """Chạy detection một ảnh; dùng chung code path với ``detect_batch``."""
        return self.detect_batch([img], confidence=confidence, room_type=room_type)[0]

    def detect_batch(
        self,
        images: list[Image.Image],
        confidence: float,
        room_type: str | None = None,
    ) -> list[list[dict]]:
        """Detect nhiều ảnh độc lập trong cùng batch Grounding DINO.

        Ảnh được tách thành micro-batch theo device. Mỗi kết quả post-process
        vẫn có một phần tử riêng cho từng ảnh, sau đó được gộp theo image index;
        không bao giờ trộn nhãn giữa các phòng.

        Các ảnh trong một batch phải dùng chung ``room_type`` (và do đó chung
        tập prompt). Caller có room khác nhau cần tách thành các batch nhỏ hơn.
        """
        if self._model is None or self._processor is None:
            raise ModelNotReadyError("Model chưa được nạp.")
        if not images:
            return []

        import torch

        prepared_images = [self._prepare_image(img) for img in images]
        text_thresh = min(self.text_threshold, float(confidence))
        best_by_image: list[dict[str, dict]] = [{} for _ in prepared_images]
        effective_batch_size = min(self.inference_batch_size, len(prepared_images))
        total_micro_batches = (
            len(prepared_images) + effective_batch_size - 1
        ) // effective_batch_size

        # Khi caller không chỉ định phòng, dùng prompt `default` gồm toàn bộ
        # CLASS_PROMPTS. Room cụ thể chỉ giới hạn nhãn khi nó được gửi rõ ràng.
        resolved_room, prompts = self._resolve_prompts(room_type)

        with self._lock, torch.no_grad():
            for idx, prompt_text in enumerate(prompts, 1):
                for batch_number, start in enumerate(
                    range(0, len(prepared_images), effective_batch_size), 1
                ):
                    end = start + effective_batch_size
                    batch_images = prepared_images[start:end]
                    logger.info(
                        "[Grounding DINO] [%s] Micro-batch %d/%d, size %d, prompt (%d/%d): %s",
                        resolved_room,
                        batch_number,
                        total_micro_batches,
                        len(batch_images),
                        idx,
                        len(prompts),
                        prompt_text,
                    )
                    # Processor nhận một text prompt tương ứng cho từng ảnh.
                    # Các ảnh cùng micro-batch dùng cùng prompt chunk.
                    inputs = self._processor(
                        images=batch_images,
                        text=[prompt_text] * len(batch_images),
                        return_tensors="pt",
                        images_kwargs={"size": self.image_size},
                    ).to(self.device)
                    outputs = self._model(**inputs)

                    post_kwargs: dict = {
                        "outputs": outputs,
                        "input_ids": inputs.input_ids,
                        "text_threshold": text_thresh,
                        "target_sizes": [
                            (img.height, img.width) for img in batch_images
                        ],
                    }
                    if self._post_process_uses_threshold:
                        post_kwargs["threshold"] = float(confidence)
                    else:
                        post_kwargs["box_threshold"] = float(confidence)

                    results = self._processor.post_process_grounded_object_detection(**post_kwargs)
                    if len(results) != len(batch_images):
                        raise ArtifactError(
                            "Grounding DINO trả số kết quả không khớp với micro-batch."
                        )
                    for local_index, result in enumerate(results):
                        raw_labels = result.get("text_labels")
                        if raw_labels is None:
                            raw_labels = result.get("labels", [])
                        scores = result.get("scores", [])

                        for raw_label, score in zip(raw_labels, scores):
                            cls_name = normalize_label(str(raw_label))
                            if cls_name is None:
                                continue
                            conf = float(score.item() if hasattr(score, "item") else score)
                            best = best_by_image[start + local_index]
                            if cls_name not in best or conf > best[cls_name]["confidence"]:
                                best[cls_name] = {"label": cls_name, "confidence": conf}

        detected_by_image = [list(best.values()) for best in best_by_image]
        for image_index, detected in enumerate(detected_by_image, 1):
            if detected:
                logger.info(
                    "[Grounding DINO] [%s] Image %d/%d: detected %d objects: %s",
                    resolved_room,
                    image_index,
                    len(detected_by_image),
                    len(detected),
                    ", ".join(f"{d['label']} ({d['confidence']:.2f})" for d in detected),
                )
            else:
                logger.info(
                    "[Grounding DINO] [%s] Image %d/%d: no object over confidence=%.2f",
                    resolved_room,
                    image_index,
                    len(detected_by_image),
                    confidence,
                )
        return detected_by_image
