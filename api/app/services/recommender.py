"""Tầng xếp hạng gợi ý: P_cooc(c | D) + W(c, r), rồi gắn sản phẩm.

Đơn vị xếp hạng là **group_code** (`tv`, `speaker`, `air_conditioner`, ...) — trùng
`mwg.product.category_id`. Tên hiển thị tiếng Việt lấy từ `group_scores.GROUP_NAMES`.

Công thức giữ nguyên notebook v5 (Cell 10.4 / 10.9 / 10.10):

    P̂(d_target | d_source) = (N(d_source ∩ d_target) + α·P(d_target)) / (N(d_source) + α·K)
    P_cooc(c | D)          = 1 - Π_{d∈D} (1 - P̂(π(c) | d))
    final_score            = P_cooc(c | D) + W(c, r)

Nguồn dữ liệu (không có bảng dự phòng trong code):

1. `W(c, r)` — `artifacts/matrix/category_room_scores.json`, nạp bởi
   `app/services/room_scores.py` rồi ghép tên -> mã qua `mwg.categories`.
2. Ma trận đồng xuất hiện — `artifacts/matrix/cooccurrence_matrix.json`.
3. Sản phẩm — PostgreSQL schema `mwg` (`catalog.products`).

`GROUP_TO_DETECTOR` phủ 22 nhóm: nhóm không có detector proxy thì `P_cooc = 0`
và xếp hạng hoàn toàn bằng `W(c, r)` — giống cách notebook xử lý
`GROUPS_WITHOUT_DETECTOR`.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from app.services import group_scores
from app.services.product_catalog import Catalog
from app.services.room_scores import ROOM_CODES

logger = logging.getLogger("app.recommender")

# ---------------------------------------------------------------------------
# Nhóm catalog -> class detector (Cell 10.2).
#
# Detector của repo này là RT-DETR huấn luyện trên ảnh nội thất COCO (45 class:
# TV, Speaker, Laptop, Monitor, Fan, Air Conditioner, ...), KHÔNG phải detector
# đồ điện máy. Vì vậy chỉ nhóm có proxy hợp lý mới được cộng P_cooc; phần còn
# lại dựa hoàn toàn vào W(c, r). Mọi giá trị dưới đây đều có trong
# `artifacts/matrix/cooccurrence_matrix.json`.
# ---------------------------------------------------------------------------
GROUP_TO_DETECTOR: dict[str, str] = {
    # Điện tử
    "tv": "TV",
    "projector": "TV",
    "speaker": "Speaker",
    "karaoke_speaker": "Speaker",
    "monitor": "Monitor",
    "laptop": "Laptop",
    "desktop_pc": "Laptop",
    "game_console": "Entertainment Center",
    "indoor_camera": "Remote",
    "outdoor_camera": "Remote",
    # Điện lạnh & gia dụng
    "air_conditioner": "Air Conditioner",
    "ceiling_fan": "Fan",
    "fan": "Fan",
    "air_cooler": "Fan",
    "heater_fan": "Fan",
    # Chiếu sáng & lưu trữ trong nhà
    "lamp": "Floor Lamp",
    "led_bulb": "Light",
    "storage_box": "Organizer Storage",
    "storage_rack": "Shelf",
    "trash_bin": "Basket/Bin",
    "power_strip": "Power Surge Protector",
}

# Detector -> nhóm cần loại khỏi gợi ý vì đã có trong ảnh (Cell 10.2).
DETECTOR_TO_GROUP: dict[str, list[str]] = {
    "TV": ["tv", "projector"],
    "Laptop": ["laptop", "desktop_pc"],
    "Monitor": ["monitor"],
    "Air Conditioner": ["air_conditioner"],
    "Speaker": ["speaker", "karaoke_speaker"],
    "Power Surge Protector": ["power_strip"],
    "Shelf": ["storage_rack"],
    "Basket/Bin": ["trash_bin"],
    "Fan": ["fan", "ceiling_fan", "air_cooler", "heater_fan"],
    "Floor Lamp": ["lamp"],
    "Light": ["led_bulb"],
    "Organizer Storage": ["storage_box"],
    "Remote": ["indoor_camera", "outdoor_camera"],
    "Entertainment Center": ["game_console"],
    # Thiết bị & gia dụng Nhà bếp
    "Refrigerator": ["refrigerator"],
    "Freezer": ["freezer"],
    "Beverage Cooler": ["beverage_cooler"],
    "Gas Stove": ["gas_stove"],
    "Induction Cooker": ["induction_cooker"],
    "Infrared Cooker": ["infrared_cooker"],
    "Microwave": ["microwave"],
    "Oven": ["oven"],
    "Range Hood": ["range_hood"],
    "Dishwasher": ["dishwasher"],
    "Rice Cooker": ["rice_cooker"],
    "Air Fryer": ["air_fryer"],
    "Kettle": ["kettle"],
    "Blender": ["blender"],
    "Juicer": ["juicer"],
    "Soy Milk Maker": ["soy_milk_maker"],
    "Coffee Maker": ["coffee_maker"],
    "Water Purifier": ["water_purifier"],
    "Water Dispenser": ["water_dispenser"],
}

# Tín hiệu bố cục có tính phân biệt phòng. Các nhãn chung như ``Chair`` chỉ có
# trọng số thấp; các vật thể neo như ``Bed`` hoặc ``Refrigerator`` mới quyết
# định mạnh loại phòng. Điểm cuối còn được bổ sung từ W(c, r) của category đã
# detect, xem ``infer_room_type``.
KITCHEN_ANCHOR_LABELS = frozenset({
    "Refrigerator", "Freezer", "Beverage Cooler", "Gas Stove", "Induction Cooker",
    "Infrared Cooker", "Microwave", "Oven", "Range Hood", "Dishwasher", "Rice Cooker",
    "Air Fryer", "Kettle", "Blender", "Juicer", "Soy Milk Maker", "Coffee Maker",
    "Water Purifier", "Water Dispenser",
})

ROOM_LABEL_WEIGHTS: dict[str, dict[str, float]] = {
    **{label: {"kitchen": 4.0} for label in KITCHEN_ANCHOR_LABELS},
    "Bed": {"bedroom": 4.0},
    "Pillow": {"bedroom": 1.5},
    "Throw Blanket": {"bedroom": 1.5},
    "Side Table": {"bedroom": 1.2},
    "Sofa": {"living_room": 3.5},
    "Coffee Table": {"living_room": 2.5},
    "Entertainment Center": {"living_room": 2.0},
    "TV": {"living_room": 1.7, "bedroom": 0.4},
    "Speaker": {"living_room": 1.2},
    "Fireplace": {"living_room": 2.5},
    "Laptop": {"bedroom": 0.5},
    "Monitor": {"bedroom": 0.4},
    "Printer": {"bedroom": 0.3},
    "Desk Lamp": {"bedroom": 0.5},
    "Tableware": {"kitchen": 1.0},
    "Mugs Teacups": {"kitchen": 0.8},
    "Table": {"kitchen": 0.3},
    "Chair": {"living_room": 0.15, "bedroom": 0.15},
    "Cabinet": {"kitchen": 0.5, "bedroom": 0.2},
    "Mirror": {"bedroom": 0.5},
    "Rug": {"living_room": 0.4, "bedroom": 0.4},
    "Plant": {"living_room": 0.2, "bedroom": 0.1},
    "Planter": {"living_room": 0.2, "bedroom": 0.1},
}

# W(c, r) được học ở cấp category, nên là tín hiệu phụ trợ; không được phép
# lấn át vật thể neo trong bố cục ảnh.
CATALOG_ROOM_SCORE_WEIGHT = 0.75
NO_SIGNAL_FALLBACK_ROOM = "living_room"

# Nhóm không có detector tương ứng — P_cooc luôn 0, chỉ xếp theo W(c, r).
GROUPS_WITHOUT_DETECTOR: frozenset[str] = frozenset(
    set(group_scores.GROUP_NAMES) - set(GROUP_TO_DETECTOR)
)


class Recommender:
    """Bất biến sau khi tạo, an toàn khi dùng đa luồng."""

    def __init__(
        self,
        cooc: Mapping[str, Any],
        catalog: Catalog,
        *,
        room_scores: Mapping[str, Mapping[str, float]],
        alpha: float = 1.0,
        smooth_k: int = 2,
        default_room_type: str = "living_room",
        default_top_k: int = 10,
        default_min_score: float = 0.05,
        default_products_per_group: int = 3,
    ):
        self.classes: list[str] = list(cooc["classes"])
        self.matrix: list[list[float]] = cooc["matrix"]
        self.support: dict[str, int] = cooc["support"]
        self.counts: dict[str, dict[str, int]] = cooc["counts"]
        self.base_rate: dict[str, float] = cooc["base_rate"]
        self.n_images: int = cooc["n_images"]

        self.catalog = catalog
        # W(c, r) do `bootstrap` nạp từ artifact rồi truyền vào — Recommender
        # không tự đọc file/database để dùng lại được với bảng khác trong test.
        self.room_scores: dict[str, dict[str, float]] = {
            code: dict(scores) for code, scores in room_scores.items()
        }
        self.alpha = alpha
        self.smooth_k = smooth_k
        self.default_room_type = default_room_type
        self.default_top_k = default_top_k
        self.default_min_score = default_min_score
        self.default_products_per_group = default_products_per_group
        self._index = {c: i for i, c in enumerate(self.classes)}

        # Ứng viên = nhóm có trong catalog VÀ có điểm phòng; tính một lần.
        self.candidate_groups: list[str] = sorted(
            code for code in self.room_scores if code in catalog.groups
        )
        missing = sorted(set(self.catalog.groups) - set(self.room_scores))
        if missing:
            logger.warning(
                "Nhóm có sản phẩm nhưng chưa có W(c, r) nên không được gợi ý: %s",
                ", ".join(missing),
                extra={"event": "groups_without_scores"},
            )

    @property
    def room_types(self) -> list[str]:
        return list(ROOM_CODES)

    # ---- Tầng 1: P̂(d_target | d_source) ----
    def p_smoothed(
        self, d_source: str, d_target: str, alpha: float | None = None, k: int | None = None
    ) -> float:
        """P̂(d_target | d_source) = (N(d_source ∩ d_target) + α·P(d_target)) / (N(d_source) + α·K)."""
        alpha = self.alpha if alpha is None else alpha
        k = self.smooth_k if k is None else k
        if d_source not in self._index or d_target not in self._index:
            return 0.0
        n_source = self.support.get(d_source, 0)
        if n_source == 0:
            return self.base_rate.get(d_target, 0.0)
        if alpha <= 0:
            return self.matrix[self._index[d_source]][self._index[d_target]]
        n_both = self.counts[d_source][d_target]
        prior = self.base_rate.get(d_target, 0.0)
        return (n_both + alpha * prior) / (n_source + alpha * k)

    # ---- Tầng 2: W(c, r) ----
    def w(self, group_code: str, room_type: str) -> float:
        """W(c, r) — đọc bảng điểm nạp từ `artifacts/matrix/category_room_scores.json`."""
        return float(self.room_scores.get(group_code, {}).get(room_type, 0.0))

    def infer_room_type(
        self, detected_objects: Sequence[dict[str, Any]],
    ) -> tuple[str, dict[str, float]]:
        """Chấm điểm 3 loại phòng từ các vật thể Grounding DINO đã detect.

        Mỗi đóng góp được nhân với confidence của nhãn. Hàm trả về đầy đủ điểm
        để API có thể giải thích quyết định, cùng room có điểm cao nhất. Nếu ảnh
        không có nhãn mang tín hiệu phòng, giữ ``default_room_type``.
        """
        scores = {room: 0.0 for room in self.room_types}

        for item in detected_objects:
            label = str(item.get("label", ""))
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            if confidence == 0.0:
                continue

            for room, weight in ROOM_LABEL_WEIGHTS.get(label, {}).items():
                scores[room] += confidence * weight

            # Các thiết bị có category tương ứng dùng thêm W(c, r), nhờ vậy
            # điểm phòng đã huấn luyện được tham gia mà không cần hard-code
            # toàn bộ danh mục vào bảng tín hiệu phía trên.
            for group_code in DETECTOR_TO_GROUP.get(label, ()):
                for room in scores:
                    scores[room] += (
                        confidence * CATALOG_ROOM_SCORE_WEIGHT * self.w(group_code, room)
                    )

        best_score = max(scores.values(), default=0.0)
        has_room_signal = best_score > 0.0
        room_type = (
            max(self.room_types, key=lambda room: scores[room])
            if has_room_signal
            else NO_SIGNAL_FALLBACK_ROOM
        )
        rounded_scores = {room: round(score, 4) for room, score in scores.items()}
        logger.info(
            "Room type inferred: %s; source=%s; scores=%s; detected_labels=%s",
            room_type,
            "signals" if has_room_signal else "fallback_no_detected_objects",
            rounded_scores,
            [str(item.get("label", "")) for item in detected_objects],
        )
        return room_type, rounded_scores

    # ---- Tầng 3: P_cooc(c | D) ----
    def cooccurrence_score(
        self, detected_labels: Sequence[str], group_code: str, alpha: float | None = None
    ) -> float:
        """P_cooc(c | D) = 1 - Π_{d∈D} (1 - P̂(π(c) | d)); 0 nếu nhóm không có proxy."""
        if not detected_labels:
            return 0.0
        proxy = GROUP_TO_DETECTOR.get(group_code)
        if proxy is None:
            return 0.0
        not_present = 1.0
        for label in detected_labels:
            not_present *= 1.0 - self.p_smoothed(label, proxy, alpha)
        return 1.0 - not_present

    def build_reason(
        self,
        group_code: str,
        detected_labels: Sequence[str],
        room_score: float,
        cooc_score: float = 0.0,
    ) -> str:
        """Câu giải thích: ưu tiên nêu vật thể trong ảnh nếu có đóng góp P_cooc.

        Nêu cả `cooc_score` (Noisy-OR tổng hợp trên mọi vật thể) chứ không chỉ
        `P̂(π(c) | best)` của một vật thể, để con số trong câu khớp với
        `cooccurrence_score` mà response trả về.
        """
        name = group_scores.group_name(group_code)
        proxy = GROUP_TO_DETECTOR.get(group_code)
        if not detected_labels or proxy is None:
            return f"{name} phù hợp với phòng đang xét (điểm phòng {room_score:.1f})."

        best_label, best_p = None, 0.0
        for label in detected_labels:
            if label == proxy:
                continue
            p = self.p_smoothed(label, proxy)
            if p > best_p:
                best_label, best_p = label, p

        if best_label is None or best_p <= 0 or cooc_score <= 0:
            return f"{name} phù hợp với phòng đang xét (điểm phòng {room_score:.1f})."

        return (
            f"Ảnh có {best_label} — thường đi kèm {name} "
            f"(kết hợp các vật thể: {cooc_score:.0%}); điểm phòng {room_score:.1f}."
        )

    # ---- Xếp hạng ----
    def recommend(
        self,
        detected_objects: Sequence[dict[str, Any]],
        *,
        room_type: str | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
        exclude_detected: bool = True,
        products_per_group: int | None = None,
        price_range: tuple[float, float] | None = None,
        alpha: float | None = None,
        category_id: str | None = None,
    ) -> dict[str, Any]:
        """Ghép P_cooc với W(c, r), lọc, xếp hạng, rồi gắn sản phẩm (Cell 10.10)."""
        room_type = room_type or self.default_room_type
        top_k = self.default_top_k if top_k is None else top_k
        min_score = self.default_min_score if min_score is None else min_score
        products_per_group = (
            self.default_products_per_group if products_per_group is None else products_per_group
        )

        detected_labels = [o["label"] for o in detected_objects]

        already_present: set[str] = set()
        if exclude_detected:
            for label in detected_labels:
                already_present.update(DETECTOR_TO_GROUP.get(label, ()))

        candidates: list[dict[str, Any]] = []
        for group_code in self.candidate_groups:
            if group_code in already_present:
                continue
            if (
                category_id is not None
                and group_scores.GROUP_CATEGORY.get(group_code) != category_id
            ):
                continue

            cooc = self.cooccurrence_score(detected_labels, group_code, alpha)
            room_score = self.w(group_code, room_type)
            final_score = cooc + room_score
            if final_score < min_score:
                continue

            candidates.append({
                "group_code": group_code,
                "group_name": group_scores.group_name(group_code),
                "category_id": group_scores.GROUP_CATEGORY.get(group_code),
                "cooccurrence_score": cooc,
                "room_score": room_score,
                "final_score": final_score,
                "product_count": self.catalog.counts.get(group_code, 0),
                "reason": self.build_reason(group_code, detected_labels, room_score, cooc),
            })

        # Sắp giảm dần theo `final_score`. Điểm phòng gán cứng chỉ có ~11 mức
        # (0.0 → 1.0 bước 0.1) nên rất nhiều nhóm đồng điểm; tie-break theo
        # P_cooc rồi số sản phẩm để kết quả ổn định và có ý nghĩa (nhóm nhiều
        # lựa chọn hơn đứng trước) thay vì rơi về thứ tự bảng chữ cái.
        candidates.sort(
            key=lambda c: (
                c["final_score"],
                c["cooccurrence_score"],
                c["product_count"],
                c["group_code"],
            ),
            reverse=True,
        )

        # Gắn sản phẩm rồi CHỈ nhận nhóm còn ít nhất một sản phẩm thoả lọc.
        # Nhóm có `products: []` là vô nghĩa với người dùng (nhất là khi lọc
        # `min_price`/`max_price`), nên bỏ qua và lấy tiếp ứng viên phía sau để
        # vẫn đủ `top_k`.
        top: list[dict[str, Any]] = []
        for item in candidates:
            products = list(self.catalog.available.get(item["group_code"], ()))
            if price_range is not None:
                lo, hi = price_range
                products = [p for p in products if lo <= p["price"] <= hi]
            if not products:
                continue
            # Rẻ nhất trước: lựa chọn dễ tiếp cận hiển thị trước.
            products.sort(key=lambda p: p["price"])
            top.append({
                "group_code": item["group_code"],
                "group_name": item["group_name"],
                "category_id": item["category_id"],
                "cooccurrence_score": item["cooccurrence_score"],
                "products": products[:products_per_group],
            })
            if len(top) >= top_k:
                break

        return {
            "room_type": room_type,
            "detected_objects": [dict(o) for o in detected_objects],
            "recommendations": top,
        }
