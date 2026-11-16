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
# Toàn bộ 62 classes sau khi gộp từ đồng nghĩa chỉ chiếm ~170 BERT tokens (< 256),
# nằm gọn trong 1 chunk duy nhất, giúp giảm 50% số lượt forward pass trên CPU.
DEFAULT_CHUNK_SIZE = 64
CROSS_CHUNK_IOU_THRESHOLD = 0.65
ANCHOR_CONFIDENCE_FLOOR = 0.42
ANCHOR_CLASSES: frozenset[str] = frozenset({
    "Bed",
    "Sofa",
    "Refrigerator",
    "Gas Stove",
    "Electric Cooktop",
    "Range Hood",
})

# Các class đã được gộp thẳng vào class chuẩn khác (Monitor -> TV;
# Induction Cooker / Infrared Cooker -> Electric Cooktop).
MERGED_CLASSES: dict[str, str] = {
    "Monitor": "TV",
    "Induction Cooker": "Electric Cooktop",
    "Infrared Cooker": "Electric Cooktop",
}

# Các class đồ trang trí/vật dụng nhỏ trong ma trận COCO không có trọng số phòng
# và không ánh xạ tới sản phẩm ĐMX, được lược bỏ khỏi prompt Grounding DINO.
PRUNED_MATRIX_CLASSES: frozenset[str] = frozenset({
    "Books",
    "Candle",
    "Clock",
    "Curtain Rod",
    "Decor",
    "Phone",
    "Stool",
    "Table Lamp",
    "Vase",
})

# ---------------------------------------------------------------------------
# Ánh xạ class chuẩn của detector -> cụm từ prompt tiếng Anh rút gọn cho Grounding DINO.
# Mỗi class chỉ giữ 1–2 cụm từ đại diện ngắn gọn nhất để toàn bộ prompt nằm trong
# 1 lượt forward pass (< 256 tokens BERT); các từ đồng nghĩa mở rộng được đặt ở
# `EXTRA_LABEL_ALIASES` để `normalize_label` vẫn quy đổi đầy đủ.
# ---------------------------------------------------------------------------
CLASS_PROMPTS: dict[str, tuple[str, ...]] = {
    # --- 35 class nội thất cơ bản (giữ Curtain, Picture Frame, Wall Art; bỏ 9 class trang trí nhỏ) ---
    "Accent Chair": ("armchair",),
    "Air Conditioner": ("air conditioner",),
    "Basket/Bin": ("trash bin", "basket"),
    "Bed": ("bed",),
    "Bench/Ottoman": ("ottoman",),
    "Cabinet": ("cabinet",),
    "Chair": ("chair",),
    "Coffee Table": ("coffee table",),
    "Curtain": ("curtain",),
    "Desk Lamp": ("desk lamp",),
    "Entertainment Center": ("tv stand",),
    "Fan": ("fan",),
    "Fireplace": ("fireplace",),
    "Floor Lamp": ("floor lamp",),
    "Laptop": ("laptop",),
    "Light": ("ceiling light",),
    "Mirror": ("mirror",),
    "Mugs Teacups": ("mug",),
    "Organizer Storage": ("storage box",),
    "Picture Frame": ("picture frame",),
    "Pillow": ("pillow",),
    "Plant": ("plant",),
    "Planter": ("planter",),
    "Power Surge Protector": ("power strip",),
    "Printer": ("printer",),
    "Remote": ("remote",),
    "Rug": ("rug",),
    "Shelf": ("shelf",),
    "Side Table": ("nightstand", "side table"),
    "Sofa": ("sofa",),
    "Speaker": ("speaker",),
    "TV": ("tv", "monitor"),
    "Table": ("dining table",),
    "Tableware": ("tableware",),
    "Throw Blanket": ("blanket",),
    "Wall Art": ("wall art",),

    # --- 18 Danh mục thiết bị & gia dụng Nhà bếp (gộp bếp điện vào Electric Cooktop) ---
    "Refrigerator": ("refrigerator",),
    "Freezer": ("freezer",),
    "Beverage Cooler": ("beverage cooler", "wine cooler"),
    "Gas Stove": ("gas stove",),
    "Electric Cooktop": ("electric cooktop", "induction stove"),
    "Microwave": ("microwave",),
    "Oven": ("oven",),
    "Range Hood": ("range hood",),
    "Dishwasher": ("dishwasher",),
    "Rice Cooker": ("rice cooker",),
    "Air Fryer": ("air fryer",),
    "Kettle": ("kettle",),
    "Blender": ("blender",),
    "Juicer": ("juicer",),
    "Soy Milk Maker": ("soy milk maker",),
    "Coffee Maker": ("coffee maker",),
    "Water Purifier": ("water purifier",),
    "Water Dispenser": ("water dispenser",),
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
    "Cabinet",
    "Chair",
    "Coffee Table",
    "Curtain",
    "Desk Lamp",
    "Entertainment Center",
    "Fan",
    "Fireplace",
    "Floor Lamp",
    "Laptop",
    "Light",
    "Mirror",
    "Mugs Teacups",
    "Organizer Storage",
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
    "TV",
    "Table",
    "Tableware",
    "Throw Blanket",
    "Wall Art",
    # Tiện ích gia dụng có thể đặt ở phòng khách
    "Beverage Cooler",
    "Coffee Maker",
    "Water Dispenser",
)

KITCHEN_CLASSES: tuple[str, ...] = (
    # Nhãn thiết bị & gia dụng nhà bếp. Electric Cooktop là nhãn detector tổng
    # quát, ánh xạ về hai category bếp điện hiện có trong recommender.
    "Refrigerator",
    "Freezer",
    "Beverage Cooler",
    "Gas Stove",
    "Electric Cooktop",
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
)

BEDROOM_CLASSES: tuple[str, ...] = (
    # Vật thể neo bố cục phòng ngủ.
    "Bed",
    "Pillow",
    "Throw Blanket",
    "Side Table",
    "Desk Lamp",
    "Floor Lamp",
    "Mirror",
    "Cabinet",
    "Shelf",
    "Rug",
    "Curtain",
    "Chair",
    "Table",
    "Picture Frame",
    "Wall Art",
    "Plant",
    "Planter",
    "Organizer Storage",
    "Laptop",
    "TV",
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


EXTRA_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "Accent Chair": ("accent chair",),
    "Basket/Bin": ("storage bin", "trash can", "garbage can"),
    "Bed": ("bed frame", "headboard"),
    "Bench/Ottoman": ("bench",),
    "Cabinet": ("dresser", "kitchen cabinet", "cupboard", "kitchen counter"),
    "Chair": ("dining chair",),
    "Entertainment Center": ("entertainment center",),
    "Fan": ("ceiling fan",),
    "Light": ("light fixture",),
    "Mugs Teacups": ("tea cup",),
    "Organizer Storage": ("organizer",),
    "Pillow": ("throw pillow",),
    "Plant": ("potted plant",),
    "Planter": ("flower pot",),
    "Power Surge Protector": ("surge protector",),
    "Remote": ("remote control",),
    "Rug": ("carpet",),
    "Shelf": ("bookshelf",),
    "Sofa": ("couch",),
    "Speaker": ("soundbar",),
    "TV": ("television", "monitor"),
    "Table": ("table",),
    "Tableware": ("bowl", "serving tray"),
    "Throw Blanket": ("throw blanket",),
    "Wall Art": ("painting",),
    "Refrigerator": ("fridge",),
    "Freezer": ("chest freezer",),
    "Beverage Cooler": ("drink cooler",),
    "Gas Stove": ("gas cooker", "gas range"),
    "Electric Cooktop": (
        "induction cooker",
        "infrared cooker",
        "infrared stove",
        "cooktop",
        "electric stove",
        "electric hob",
        "built-in electric cooktop",
        "glass ceramic cooktop",
        "ceramic hob",
        "electric hot plate",
        "single burner electric cooktop",
    ),
    "Microwave": ("microwave oven",),
    "Oven": ("electric oven",),
    "Range Hood": ("kitchen hood", "exhaust hood"),
    "Kettle": ("electric kettle", "water kettle"),
    "Blender": ("smoothie maker",),
    "Juicer": ("fruit juicer",),
    "Soy Milk Maker": ("nut milk maker",),
    "Coffee Maker": ("coffee machine", "espresso machine"),
    "Water Purifier": ("water filter",),
    "Water Dispenser": ("water cooler",),
}


def _box_iou(box_a: list[float] | tuple[float, ...], box_b: list[float] | tuple[float, ...]) -> float:
    """Tính IoU giữa 2 bounding box [x1, y1, x2, y2]."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _build_lookup() -> tuple[dict[str, str], list[tuple[str, str]]]:
    exact: dict[str, str] = {}
    for canonical, phrases in CLASS_PROMPTS.items():
        exact[canonical.lower()] = canonical
        exact[_clean_text(canonical)] = canonical
        for phrase in phrases:
            exact[phrase.lower()] = canonical
            exact[_clean_text(phrase)] = canonical
    for merged_src, merged_target in MERGED_CLASSES.items():
        exact[merged_src.lower()] = merged_target
        exact[_clean_text(merged_src)] = merged_target
    for canonical, aliases in EXTRA_LABEL_ALIASES.items():
        for alias in aliases:
            exact[alias.lower()] = canonical
            exact[_clean_text(alias)] = canonical

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
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[str]:
    """Tách danh sách class thành các chuỗi prompt < 240 tokens để an toàn với giới hạn 256 của BERT."""
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
    """Bọc Grounding DINO; một model dùng chung nên inference được tuần tự hoá."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_ID,
        device: str = "auto",
        jpeg_roundtrip: bool = False,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        shortest_edge: int = 320,
        longest_edge: int = 512,
    ):
        self.model_path = Path(model_path) if isinstance(model_path, Path) else Path(str(model_path))
        self.model_id = str(model_path).strip() or DEFAULT_MODEL_ID
        self.device = resolve_device(device)
        self.jpeg_roundtrip = jpeg_roundtrip
        self.text_threshold = text_threshold
        self.image_size = {
            "shortest_edge": shortest_edge,
            "longest_edge": longest_edge,
        }
        self.family: str | None = None
        self._processor = None
        self._model = None
        self._classes: list[str] = list(CLASS_PROMPTS.keys())
        self.room_prompts: dict[str, list[str]] = {
            room: build_chunked_prompts(classes, chunk_size=DEFAULT_CHUNK_SIZE)
            for room, classes in ROOM_CLASSES.items()
        }
        self.room_prompts["default"] = build_chunked_prompts(self._classes, chunk_size=DEFAULT_CHUNK_SIZE)
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
            import os
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            if self.device == "cpu":
                cpu_cores = os.cpu_count() or 4
                # Trên Apple Silicon / multi-core hỗn hợp (P-cores + E-cores),
                # giới hạn số luồng tránh nghẽn OpenMP barrier trên các nhân E-core chậm.
                optimal_threads = 8 if cpu_cores >= 10 else max(1, cpu_cores)
                torch.set_num_threads(optimal_threads)

            self._processor = AutoProcessor.from_pretrained(target)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(target)
            self._model.to(self.device)
            self._model.eval()

            # Bọc cache cho BERT text_backbone vì các chuỗi prompt theo phòng là cố định.
            base_model = getattr(self._model, "model", None)
            orig_tb = getattr(base_model, "text_backbone", None)
            if orig_tb is not None:
                tb_cache: dict[tuple[int, ...], object] = {}

                class _CachedTextBackbone(torch.nn.Module):
                    def __init__(self, inner: torch.nn.Module) -> None:
                        super().__init__()
                        self.inner = inner

                    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, **kwargs):
                        if input_ids is not None and input_ids.shape[0] == 1:
                            key = tuple(input_ids[0].tolist())
                            cached = tb_cache.get(key)
                            if cached is not None:
                                return cached
                            out = self.inner(
                                input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                **kwargs,
                            )
                            if len(tb_cache) < 16:
                                tb_cache[key] = out
                            return out
                        return self.inner(
                            input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                             **kwargs,
                        )

                base_model.text_backbone = _CachedTextBackbone(orig_tb)

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
        """Bộ class của detector (cùng các class đã gộp/lược bỏ) phải bao phủ đầy đủ ma trận."""
        actual = set(self.class_names) | set(MERGED_CLASSES.keys()) | set(PRUNED_MATRIX_CLASSES)
        missing = sorted(set(expected) - actual)
        if missing:
            raise ArtifactError(
                "Class của detector thiếu so với ma trận đồng xuất hiện: "
                f"(thiếu {missing[:5]})."
            )

    def warmup(self) -> None:
        if self.loaded:
            dummy = Image.new("RGB", (64, 64))
            self.detect(dummy, confidence=0.99, room_type=None)
            self.detect(dummy, confidence=0.99, room_type="living_room")

    def _prepare_image(self, img: Image.Image) -> Image.Image:
        """Chuẩn hoá đầu vào ảnh trước khi đưa vào Grounding DINO."""
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

    def detect(
        self,
        img: Image.Image,
        confidence: float,
        room_type: str | None = None,
    ) -> list[dict]:
        """Chạy Grounding DINO cho một ảnh và quy đổi về các class chuẩn."""
        if self._model is None or self._processor is None:
            raise ModelNotReadyError("Model chưa được nạp.")

        import torch

        prepared_image = self._prepare_image(img)
        req_conf = float(confidence)
        # Với vật thể neo phòng diện tích lớn (Bed, Sofa, Refrigerator...),
        # cho phép ngưỡng sàn 0.42 khi caller truyền confidence=0.50 để tránh rớt sát nút.
        query_thresh = min(req_conf, ANCHOR_CONFIDENCE_FLOOR) if req_conf < 0.90 else req_conf
        text_thresh = max(self.text_threshold, query_thresh - 0.05) if req_conf < 0.90 else min(self.text_threshold, req_conf)
        candidates: list[dict] = []

        # Khi caller không chỉ định phòng, dùng prompt `default` gồm toàn bộ
        # CLASS_PROMPTS. Room cụ thể chỉ giới hạn nhãn khi nó được gửi rõ ràng.
        resolved_room, prompts = self._resolve_prompts(room_type)

        with self._lock, torch.no_grad():
            for idx, prompt_text in enumerate(prompts, 1):
                logger.info(
                    "[Grounding DINO] [%s] Text prompt (%d/%d): %s",
                    resolved_room,
                    idx,
                    len(prompts),
                    prompt_text,
                )
                inputs = self._processor(
                    images=prepared_image,
                    text=prompt_text,
                    return_tensors="pt",
                    images_kwargs={"size": self.image_size},
                ).to(self.device)
                outputs = self._model(**inputs)

                post_kwargs: dict = {
                    "outputs": outputs,
                    "input_ids": inputs.input_ids,
                    "text_threshold": text_thresh,
                    "target_sizes": [(prepared_image.height, prepared_image.width)],
                }
                if self._post_process_uses_threshold:
                    post_kwargs["threshold"] = query_thresh
                else:
                    post_kwargs["box_threshold"] = query_thresh

                results = self._processor.post_process_grounded_object_detection(**post_kwargs)
                result = results[0]
                raw_labels = result.get("text_labels")
                if raw_labels is None:
                    raw_labels = result.get("labels", [])
                scores = result.get("scores", [])
                boxes = result.get("boxes", [])

                for box_idx, (raw_label, score) in enumerate(zip(raw_labels, scores)):
                    cls_name = normalize_label(str(raw_label))
                    if cls_name is None:
                        continue
                    conf = float(score.item() if hasattr(score, "item") else score)
                    min_needed = query_thresh if cls_name in ANCHOR_CLASSES else req_conf
                    if conf < min_needed:
                        continue
                    raw_box = boxes[box_idx] if box_idx < len(boxes) else None
                    if raw_box is not None:
                        box_coords = [
                            float(x.item() if hasattr(x, "item") else x)
                            for x in raw_box
                        ]
                    else:
                        box_coords = None
                    candidates.append({
                        "label": cls_name,
                        "confidence": conf,
                        "box": box_coords,
                    })

        # Cross-chunk Box NMS: nếu cùng một vật thể (IoU >= 0.65) bị gắn 2 nhãn
        # khác nhau ở 2 chunks (vd: Air Conditioner 0.85 ở chunk 1 và Air Fryer 0.58
        # ở chunk 2), chỉ giữ lại nhãn có confidence cao hơn.
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        kept_boxes: list[dict] = []
        best: dict[str, dict] = {}
        for cand in candidates:
            cand_box = cand["box"]
            if cand_box is not None and any(
                kb["box"] is not None
                and kb["label"] != cand["label"]
                and _box_iou(cand_box, kb["box"]) >= CROSS_CHUNK_IOU_THRESHOLD
                for kb in kept_boxes
            ):
                continue
            kept_boxes.append(cand)
            cls_name = cand["label"]
            conf = cand["confidence"]
            if cls_name not in best or conf > best[cls_name]["confidence"]:
                best[cls_name] = {"label": cls_name, "confidence": conf}

        detected = list(best.values())
        if detected:
            logger.info(
                "[Grounding DINO] [%s] Detected %d objects: %s",
                resolved_room,
                len(detected),
                ", ".join(f"{d['label']} ({d['confidence']:.2f})" for d in detected),
            )
        else:
            logger.info(
                "[Grounding DINO] [%s] No object over confidence=%.2f",
                resolved_room,
                confidence,
            )
        return detected
