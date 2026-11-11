"""Module xây dựng Prompt chuyên biệt cho Qwen Image Edit Gateway."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger("app.prompt_builder")


# Các bước đọc bố cục chỉ thêm cho đúng loại phòng. `living_room` giữ nguyên
# prompt hiện có để không làm thay đổi hành vi đang được sử dụng.
ROOM_SCENE_ANALYSIS: dict[str, str] = {
    "kitchen": (
        "Kitchen Scene Analysis: Before placing any product, identify the existing walls, floor, kitchen cabinets, "
        "countertops, kitchen shelves, sink, stove, doors, appliances, and all existing kitchen objects in picture 1. "
        "Preserve their positions and use only a suitable existing floor area, counter, shelf, or wall plane."
    ),
    "bedroom": (
        "Bedroom Scene Analysis: Before placing any product, identify the existing bed, floor, bedside tables, desk, chairs, "
        "mirror, wardrobe, windows, lamps, wall decor, and all existing bedroom objects in picture 1. "
        "Preserve their positions and use only a suitable existing floor area, tabletop, shelf, or wall plane."
    ),
}


@dataclass(frozen=True)
class ProductPlacementSpec:
    """Đặc tả vị trí cho một sản phẩm trong câu prompt."""

    directive: str
    rule: str | None = None
    negative_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class CategoryRuleConfig:
    """Cấu hình quy tắc vị trí cho một nhóm sản phẩm."""

    directive_template: str
    rule: str
    negative_terms: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Bảng quy tắc chuyên biệt cho toàn bộ 62 nhóm trong mwg.categories & category_room_scores
# Quy tắc nghiêm ngặt: Luôn dùng "existing" và CẤM sinh thêm bàn/kệ mới.
# ---------------------------------------------------------------------------
CATEGORY_RULES: dict[str, CategoryRuleConfig] = {
    # 1. Bếp gas
    "gas_stove": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the gas stove resting flat on an existing kitchen counter in picture 1. "
            "Do NOT add or invent any new tables, counters, or cabinets."
        ),
        rule="Gas Stove Placement: Must sit flat on an existing counter surface in picture 1. Never on floor.",
        negative_terms=("gas stove on floor", "floating gas stove", "wall mounted stove", "extra table"),
        aliases=("gas_stove", "bếp gas", "bep gas", "gas stove"),
    ),
    # 2. Bếp từ
    "induction_cooker": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the induction cooker resting flat on an existing counter, table, or shelf in picture 1. "
            "Do NOT add any new tables, desks, or furniture."
        ),
        rule="Induction Cooker Placement: Must rest flat on an existing surface in picture 1. Never add new tables or mount on walls.",
        negative_terms=("induction cooker on floor", "floating induction cooker", "extra table"),
        aliases=("induction_cooker", "bếp từ", "bep tu", "induction cooker", "induction stove"),
    ),
    # 3. Bếp hồng ngoại
    "infrared_cooker": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the infrared cooker resting flat on an existing counter or table in picture 1. "
            "Do NOT add any new tables or furniture."
        ),
        rule="Infrared Cooker Placement: Must rest flat on an existing surface in picture 1. Never add new tables or mount on walls.",
        negative_terms=("infrared cooker on floor", "floating infrared cooker", "extra table"),
        aliases=("infrared_cooker", "bếp hồng ngoại", "bep hong ngoai", "infrared cooker", "infrared stove"),
    ),
    # 4. Tủ lạnh
    "refrigerator": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the refrigerator upright on the existing floor against an existing wall in picture 1. "
            "Position it in the left or right end bay of the existing kitchen cabinet run, with one refrigerator side flush and directly adjacent to the cabinet end panel, "
            "and its back flush against the wall. Match the camera perspective, vanishing lines, and viewing angle of the adjacent cabinet faces: orient the refrigerator parallel to the nearest cabinet or wall plane, never artificially front-facing to the camera. "
            "Partial occlusion by an existing cabinet end panel, counter edge, wall, or the image-frame edge is acceptable and preferred over moving or altering existing cabinetry; do not force the entire refrigerator to be visible. "
            "Preserve a clear walkway and do not block the counter, sink, oven, or cabinet doors. Do NOT place it in the middle of the kitchen, on tables, or on counters."
        ),
        rule=(
            "Refrigerator Placement: Stand upright on the existing floor, back flush against an existing wall and one side directly adjacent to the end of the existing kitchen cabinets. "
            "Match the adjacent cabinet perspective and allow natural partial occlusion by existing architecture or cabinetry; do not rotate it to face the camera. Keep it outside the walkway; never float or block cabinets."
        ),
        negative_terms=(
            "floating refrigerator",
            "refrigerator on table",
            "refrigerator on counter",
            "tilted refrigerator",
            "mini refrigerator on counter",
            "refrigerator in middle of kitchen",
            "refrigerator blocking walkway",
            "refrigerator blocking cabinet doors",
            "refrigerator blocking sink",
            "refrigerator away from wall",
            "refrigerator front facing camera",
            "refrigerator wrong perspective",
            "refrigerator forced fully visible",
        ),
        aliases=("refrigerator", "tủ lạnh", "tu lanh", "fridge"),
    ),
    # 5. Lò nướng
    "oven": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the electric oven resting stably on an existing counter or existing shelf in picture 1. "
            "Countertop scale (~35cm tall, ~50cm wide). Do NOT place on floor. Do NOT add any new tables or cabinets."
        ),
        rule="Oven Placement: Must rest flat on an existing counter or shelf surface. Never place on floor or invent new furniture.",
        negative_terms=("oven on floor", "floating oven", "extra table", "giant oven", "oversized oven"),
        aliases=("oven", "lò nướng", "lo nuong", "electric oven"),
    ),
    # 6. Lò vi sóng
    "microwave": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the microwave oven resting flat on an existing counter or existing shelf in picture 1. "
            "Countertop scale (~30cm tall, ~50cm wide). Do NOT add any new tables or cabinets."
        ),
        rule="Microwave Placement: Must sit flat on an existing counter or shelf, never on the floor and do not add new tables.",
        negative_terms=("microwave on floor", "floating microwave", "extra table", "giant microwave", "oversized microwave"),
        aliases=("microwave", "lò vi sóng", "lo vi song", "microwave oven"),
    ),
    # 7. Máy điều hòa
    "air_conditioner": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Mount high on the existing upper wall plane near the ceiling in picture 1. "
            "NEVER on floor, shelves, cabinets, or console. Do NOT modify the wall or ceiling."
        ),
        rule="Air Conditioner Placement: Mount high on the existing upper wall plane near the ceiling, alway higher than tivi. Keep wall unchanged.",
        negative_terms=(
            "air conditioner on floor",
            "air conditioner on shelf",
            "air conditioner on cabinet",
            "air conditioner on table",
            "air conditioner on console",
            "air conditioner sitting on furniture",
            "low mounted air conditioner",
        ),
        aliases=("air_conditioner", "máy điều hòa", "máy lạnh", "may lanh", "may dieu hoa", "air conditioner"),
    ),
    # 8. Quạt điều hòa
    "air_cooler": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the air cooler standing upright on the existing floor in picture 1. "
            "Do NOT place on tables or elevate."
        ),
        rule="Air Cooler Placement: Must stand upright with its base resting flat on the existing floor.",
        negative_terms=("floating air cooler", "air cooler on table"),
        aliases=("air_cooler", "quạt điều hòa", "quat dieu hoa", "air cooler"),
    ),
    # 9. Nồi chiên không dầu
    "air_fryer": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place one compact air fryer resting flat on an existing counter or shelf in picture 1. "
            "Small tabletop scale (~30cm tall). Never place on floor; do NOT add new tables."
        ),
        rule="Air Fryer Placement: Must rest flat on an existing counter or shelf surface. Never place on the floor.",
        negative_terms=("air fryer on floor", "floating air fryer", "extra table", "giant air fryer", "oversized air fryer"),
        aliases=("air_fryer", "nồi chiên không dầu", "noi chien khong dau", "air fryer"),
    ),
    # 10. Máy lọc không khí
    "air_purifier": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the air purifier standing upright on the existing floor near a corner or existing furniture in picture 1."
        ),
        rule="Air Purifier Placement: Must stand upright on the existing floor, never mounted high on walls or floating.",
        negative_terms=("floating air purifier", "air purifier mounted high on wall"),
        aliases=("air_purifier", "máy lọc không khí", "may loc khong khi", "air purifier"),
    ),
    # 11. Tủ mát
    "beverage_cooler": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the beverage cooler standing upright on the existing floor against the wall in picture 1."
        ),
        rule="Beverage Cooler Placement: Must stand upright with its base flat on the existing floor.",
        negative_terms=("floating cooler", "cooler on tabletop"),
        aliases=("beverage_cooler", "tủ mát", "tu mat", "beverage cooler"),
    ),
    # 12. Máy xay sinh tố
    "blender": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place one compact blender resting upright flat on an existing counter or shelf in picture 1. "
            "Small tabletop scale (~30cm tall). Never place on floor; do NOT add new tables."
        ),
        rule="Blender Placement: Must rest upright on an existing counter or shelf surface. Never place on the floor.",
        negative_terms=("blender on floor", "floating blender", "extra table", "giant blender", "oversized blender"),
        aliases=("blender", "máy xay sinh tố", "may xay sinh to", "blender"),
    ),
    # 13. Cân sức khỏe
    "body_scale": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the digital weight scale resting completely flat on the existing floor in picture 1."
        ),
        rule="Scale Placement: Must rest flat on the existing floor surface, never on tables or elevated furniture.",
        negative_terms=("floating scale", "scale on table"),
        aliases=("body_scale", "cân sức khỏe", "can suc khoe", "weight scale"),
    ),
    # 14. Quạt trần
    "ceiling_fan": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Mount securely onto the existing ceiling in picture 1. "
            "Do NOT modify the ceiling."
        ),
        rule="Ceiling Fan Placement: Must be mounted securely to the existing ceiling.",
        negative_terms=("ceiling fan on floor", "ceiling fan on wall"),
        aliases=("ceiling_fan", "quạt trần", "quat tran", "ceiling fan"),
    ),
    # 15. Máy sấy quần áo
    "clothes_dryer": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the clothes dryer standing upright firmly on the existing floor in picture 1."
        ),
        rule="Clothes Dryer Placement: Must stand upright with its base flat on the existing floor.",
        negative_terms=("floating clothes dryer", "dryer on table"),
        aliases=("clothes_dryer", "máy sấy quần áo", "may say quan ao", "dryer"),
    ),
    # 16. Máy pha cà phê
    "coffee_maker": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place one compact coffee maker resting upright flat on an existing counter, desk, or shelf in picture 1. "
            "Small tabletop scale (~25-30cm tall). Never place on floor; do NOT add new tables."
        ),
        rule="Coffee Maker Placement: Must rest upright on an existing surface. Never place on floor or invent new furniture.",
        negative_terms=("coffee maker on floor", "floating coffee maker", "extra table", "giant coffee maker", "oversized coffee maker"),
        aliases=("coffee_maker", "máy pha cà phê", "may pha ca phe", "coffee maker"),
    ),
    # 17. Máy hút ẩm
    "dehumidifier": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the dehumidifier standing upright on the existing floor in picture 1."
        ),
        rule="Dehumidifier Placement: Must stand upright with its base on the existing floor.",
        negative_terms=("floating dehumidifier",),
        aliases=("dehumidifier", "máy hút ẩm", "may hut am"),
    ),
    # 18. Máy tính để bàn
    "desktop_pc": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the desktop computer case resting on an existing desk or on the existing floor beside the desk in picture 1. "
            "Do NOT add any new desks."
        ),
        rule="Desktop PC Placement: Must sit flat on an existing desk or floor.",
        negative_terms=("floating computer", "extra desk"),
        aliases=("desktop_pc", "máy tính để bàn", "may tinh de ban", "desktop pc"),
    ),
    # 19. Máy rửa chén
    "dishwasher": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the dishwasher standing on the existing kitchen floor in picture 1. "
            "Do NOT place on tables."
        ),
        rule="Dishwasher Placement: Must sit flat on the existing floor.",
        negative_terms=("dishwasher on table", "floating dishwasher"),
        aliases=("dishwasher", "máy rửa chén", "máy rửa bát", "may rua chen", "may rua bat"),
    ),
    # 20. Bàn chải đánh răng điện
    "electric_toothbrush": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the compact electric toothbrush standing upright on an existing vanity counter or shelf in picture 1. "
            "Realistic small handheld scale (~20cm tall)."
        ),
        rule="Toothbrush Placement: Must stand upright on an existing vanity surface or shelf.",
        negative_terms=("floating toothbrush", "toothbrush on floor", "giant toothbrush", "oversized toothbrush"),
        aliases=("electric_toothbrush", "bàn chải đánh răng điện", "ban chai danh rang dien"),
    ),
    # 21. Máy rửa mặt
    "facial_cleanser": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the compact facial cleanser resting on an existing vanity or shelf in picture 1. "
            "Realistic small handheld scale (~15cm)."
        ),
        rule="Facial Cleanser Placement: Must rest on an existing shelf or vanity surface.",
        negative_terms=("floating cleanser", "giant cleanser", "oversized cleanser"),
        aliases=("facial_cleanser", "máy rửa mặt", "may rua mat"),
    ),
    # 22. Quạt điện
    "fan": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the electric fan standing upright on the existing floor in picture 1."
        ),
        rule="Electric Fan Placement: Must stand upright with its base resting flat on the existing floor.",
        negative_terms=("floating fan", "fan hovering"),
        aliases=("fan", "quạt điện", "quat dien", "standing fan"),
    ),
    # 23. Bồn ngâm chân
    "foot_bath": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the foot bath massager resting flat on the existing floor in picture 1."
        ),
        rule="Foot Bath Placement: Must sit flat on the existing floor, never on tables or furniture.",
        negative_terms=("floating foot bath", "foot bath on table"),
        aliases=("foot_bath", "bồn ngâm chân", "bon ngam chan"),
    ),
    # 24. Tủ đông
    "freezer": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the freezer standing upright firmly on the existing floor along the wall in picture 1."
        ),
        rule="Freezer Placement: Must sit flat and level on the existing floor, never on tables or floating.",
        negative_terms=("floating freezer", "freezer on counter"),
        aliases=("freezer", "tủ đông", "tu dong"),
    ),
    # 25. Máy chơi game
    "game_console": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the gaming console resting flat on top of the existing TV stand or desk shelf in picture 1. "
            "Do NOT add any new tables."
        ),
        rule="Game Console Placement: Must rest flat on an existing TV console or shelf.",
        negative_terms=("floating console", "extra table"),
        aliases=("game_console", "máy chơi game", "may choi game", "console"),
    ),
    # 26. Máy sấy tóc
    "hair_dryer": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the compact hair dryer resting on an existing vanity, shelf, or tabletop in picture 1. "
            "Realistic handheld scale (~20-25cm). Do NOT add any new furniture."
        ),
        rule="Hair Dryer Placement: Must rest flat on an existing surface or shelf.",
        negative_terms=("floating hair dryer", "hair dryer on floor", "extra table", "giant hair dryer", "oversized hair dryer"),
        aliases=("hair_dryer", "máy sấy tóc", "may say toc", "hair dryer"),
    ),
    # 27. Máy tạo kiểu tóc
    "hair_styler": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the hair styling device resting on an existing vanity or tabletop in picture 1. "
            "Realistic handheld scale. Do NOT add any new furniture."
        ),
        rule="Hair Styler Placement: Must rest flat on an existing surface.",
        negative_terms=("floating styler", "styler on floor", "extra table", "giant styler"),
        aliases=("hair_styler", "máy tạo kiểu tóc", "may tao kieu toc"),
    ),
    # 28. Quạt sưởi
    "heater_fan": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the fan heater standing upright on the existing floor in picture 1."
        ),
        rule="Heater Placement: Must sit stably with its base on the existing floor.",
        negative_terms=("floating heater",),
        aliases=("heater_fan", "quạt sưởi", "quat suoi", "heater"),
    ),
    # 29. Camera trong nhà
    "indoor_camera": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the compact indoor camera resting on an existing shelf or mounted neatly on the existing wall in picture 1. "
            "Realistic small device scale (~10-15cm). Do NOT add any new tables."
        ),
        rule="Indoor Camera Placement: Must sit firmly on an existing shelf or mount flat to the existing wall plane.",
        negative_terms=("floating camera", "extra table", "giant camera", "oversized camera"),
        aliases=("indoor_camera", "camera trong nhà", "camera trong nha", "indoor camera"),
    ),
    # 30. Bàn ủi
    "iron": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the compact clothes iron resting upright on its heel on an existing surface in picture 1. "
            "Realistic handheld scale (~25cm long). Do NOT add any new tables."
        ),
        rule="Iron Placement: Must rest on an existing surface, never floating.",
        negative_terms=("floating iron", "extra table", "giant iron", "oversized iron"),
        aliases=("iron", "bàn ủi", "ban ui", "iron"),
    ),
    # 31. Máy ép trái cây
    "juicer": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place one compact juicer resting upright flat on an existing counter or shelf in picture 1. "
            "Small tabletop scale (~30cm tall). Never place on floor; do NOT add new tables."
        ),
        rule="Juicer Placement: Must rest upright on an existing counter or shelf. Never place on the floor.",
        negative_terms=("juicer on floor", "floating juicer", "extra table", "giant juicer", "oversized juicer"),
        aliases=("juicer", "máy ép trái cây", "may ep trai cay", "juicer"),
    ),
    # 32. Loa karaoke
    "karaoke_speaker": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the karaoke speaker standing upright on the existing floor in picture 1."
        ),
        rule="Karaoke Speaker Placement: Must stand upright on the existing floor.",
        negative_terms=("floating speaker", "speaker on wall"),
        aliases=("karaoke_speaker", "loa karaoke", "karaoke speaker"),
    ),
    # 33. Ấm đun nước
    "kettle": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place one compact electric kettle resting flat on an existing counter or table in picture 1. "
            "Small tabletop scale (~20cm tall, tiny footprint). Never place on floor; do NOT add new tables."
        ),
        rule="Kettle Placement & Scale: Must rest on its base flat on an existing surface at realistic compact size (~20cm tall). Never on floor.",
        negative_terms=(
            "tall kettle",
            "tall pitcher",
            "tall appliance on counter",
            "oversized on counter",
            "giant on counter",
            "kettle on floor",
            "floating kettle",
            "extra table",
            "giant kettle",
            "huge kettle",
            "oversized kettle",
            "giant appliance",
        ),
        aliases=("kettle", "ấm đun nước", "am dun nuoc", "kettle"),
    ),
    # 34. Đèn điện/đèn sạc
    "lamp": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the lamp standing on an existing desk, nightstand, or shelf in picture 1. "
            "Do NOT add any new tables or desks."
        ),
        rule="Lamp Placement: Must stand stably on an existing surface in picture 1.",
        negative_terms=("floating lamp", "lamp in mid-air", "extra table"),
        aliases=("lamp", "đèn điện", "đèn sạc", "den dien", "den sac", "desk lamp"),
    ),
    # 35. Laptop
    "laptop": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the laptop resting flat on an existing desk or table in picture 1. "
            "Do NOT add any new desks or tables."
        ),
        rule="Laptop Placement: Must rest flat on an existing tabletop or desk surface in picture 1.",
        negative_terms=("floating laptop", "extra desk", "extra table"),
        aliases=("laptop", "laptop"),
    ),
    # 36. Bóng đèn LED
    "led_bulb": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Install into the existing ceiling light fixture or lamp socket in picture 1."
        ),
        rule="LED Bulb Placement: Must be fitted into an existing light socket or fixture.",
        negative_terms=("floating bulb",),
        aliases=("led_bulb", "bóng đèn led", "bong den led", "led bulb"),
    ),
    # 37. Ghế massage
    "massage_chair": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the massage chair standing upright on the existing floor in picture 1."
        ),
        rule="Massage Chair Placement: Must sit flat and stable on the existing floor, realistically proportioned.",
        negative_terms=("floating massage chair", "tilted massage chair"),
        aliases=("massage_chair", "ghế massage", "ghe massage"),
    ),
    # 38. Thiết bị massage
    "massager": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the massage device resting on an existing sofa or chair in picture 1. "
            "Do NOT add any new furniture."
        ),
        rule="Massager Placement: Must rest stably on an existing sofa or chair surface.",
        negative_terms=("floating massager", "extra chair", "extra sofa"),
        aliases=("massager", "thiết bị massage", "thiet bi massage"),
    ),
    # 39. Microphone
    "microphone": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the microphone resting on top of the existing TV console shelf or table in picture 1. "
            "Do NOT add any new tables."
        ),
        rule="Microphone Placement: Must rest flat on an existing furniture surface.",
        negative_terms=("floating microphone", "extra table"),
        aliases=("microphone", "microphone", "micro"),
    ),
    # 40. Màn hình máy tính
    "monitor": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the computer monitor standing on an existing desk with its base stand resting firmly on the desktop surface in picture 1. "
            "Do NOT add any new desks."
        ),
        rule="Monitor Placement: Base stand must rest firmly on an existing desk surface with zero gap under the base.",
        negative_terms=("floating monitor", "gap under monitor stand", "extra desk"),
        aliases=("monitor", "màn hình máy tính", "man hinh may tinh", "monitor"),
    ),
    # 41. Bộ lau nhà
    "mop_set": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the mop and bucket set on the existing floor in the corner of the room in picture 1."
        ),
        rule="Mop Set Placement: Must rest flat on the existing floor.",
        negative_terms=("floating mop", "mop on table"),
        aliases=("mop_set", "bộ lau nhà", "bo lau nha"),
    ),
    # 42. Camera ngoài trời
    "outdoor_camera": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Mount securely onto the existing wall or balcony plane in picture 1."
        ),
        rule="Outdoor Camera Placement: Must be mounted firmly to the existing wall plane.",
        negative_terms=("outdoor camera on floor", "floating camera"),
        aliases=("outdoor_camera", "camera ngoài trời", "camera ngoai troi", "outdoor camera"),
    ),
    # 43. Ổ cắm điện
    "power_strip": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the power strip resting flat on the existing floor near the wall or on the back of an existing desk in picture 1."
        ),
        rule="Power Strip Placement: Must rest flat on the existing floor or existing desk.",
        negative_terms=("floating power strip",),
        aliases=("power_strip", "ổ cắm điện", "o cam dien", "power strip"),
    ),
    # 44. Máy chiếu
    "projector": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place compact projector flat on existing console shelf, shelf, or floor in picture 1 (~20-30cm wide). "
            "Do NOT place in foreground; do NOT add new tables or stands."
        ),
        rule="Projector Placement: Must rest on existing shelf, console, or floor. Do not add new tables.",
        negative_terms=(
            "floating projector",
            "extra table",
            "extra desk",
            "wooden table in foreground",
            "giant projector",
            "oversized projector",
            "projector in foreground",
        ),
        aliases=("projector", "máy chiếu", "may chieu", "projector"),
    ),
    # 45. Máy hút mùi
    "range_hood": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Mount on the existing wall directly above the cooking area in picture 1. "
            "Do NOT modify the wall or add furniture."
        ),
        rule="Range Hood Placement: Must be mounted on the existing wall directly above the cooking area.",
        negative_terms=("range hood on floor", "range hood on counter"),
        aliases=("range_hood", "máy hút mùi", "may hut mui", "range hood", "cooker hood"),
    ),
    # 46. Nồi cơm điện
    "rice_cooker": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place EXACTLY ONE single compact rice cooker resting flat on an existing kitchen counter in picture 1. "
            "Small tabletop scale (~20-25cm tall). Strictly 1 rice cooker only; NEVER generate multiple or duplicate rice cookers."
        ),
        rule="Rice Cooker Placement: Must place exactly ONE single rice cooker flat on an existing counter. Never duplicate into multiple rice cookers.",
        negative_terms=(
            "rice cooker on floor",
            "floating rice cooker",
            "extra table",
            "giant rice cooker",
            "oversized rice cooker",
            "duplicate rice cooker",
            "multiple rice cookers",
            "two rice cookers",
            "three rice cookers",
            "repeated rice cooker",
            "cloned rice cooker",
        ),
        aliases=("rice_cooker", "nồi cơm điện", "noi com dien", "rice cooker"),
    ),
    # 47. Robot hút bụi
    "robot_vacuum": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the robot vacuum cleaner flat on the existing floor in picture 1."
        ),
        rule="Robot Vacuum Placement: Must sit completely flat on the existing floor.",
        negative_terms=("floating robot vacuum", "robot vacuum on table"),
        aliases=("robot_vacuum", "robot hút bụi", "robot hut bui"),
    ),
    # 48. Máy cạo râu
    "shaver": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the compact shaver resting on an existing vanity or shelf in picture 1. "
            "Realistic small handheld scale (~15cm). Do NOT add any new furniture."
        ),
        rule="Shaver Placement: Must rest on an existing vanity or shelf surface.",
        negative_terms=("floating shaver", "extra table", "giant shaver", "oversized shaver"),
        aliases=("shaver", "máy cạo râu", "may cao rau", "shaver"),
    ),
    # 49. Đèn năng lượng mặt trời
    "solar_light": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Mount on the existing exterior wall or balcony railing in picture 1."
        ),
        rule="Solar Light Placement: Must be mounted securely to the existing wall or railing.",
        negative_terms=("solar light on floor",),
        aliases=("solar_light", "đèn năng lượng mặt trời", "den nang luong mat troi"),
    ),
    # 50. Máy làm sữa hạt
    "soy_milk_maker": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place one compact appliance resting flat on an existing counter or island in picture 1. "
            "Small tabletop scale (~20cm tall, tiny footprint). Never place on floor; do NOT add new tables."
        ),
        rule="Soy Milk Maker Placement: Must rest flat on existing counter/island at compact scale (~20cm tall). Never on floor.",
        negative_terms=(
            "tall kettle",
            "tall pitcher",
            "tall appliance on counter",
            "oversized on counter",
            "giant on counter",
            "appliance on floor",
            "floating appliance",
            "extra table",
            "giant appliance",
            "oversized appliance",
        ),
        aliases=("soy_milk_maker", "máy làm sữa hạt", "may lam sua hat"),
    ),
    # 51. Loa
    "speaker": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the audio speaker resting on the existing TV console, shelf, or floor in picture 1. "
            "Realistic speaker proportions. Do NOT add any new tables."
        ),
        rule="Speaker Placement: Must sit stably on an existing shelf, console, or floor.",
        negative_terms=("floating speaker", "extra table", "giant speaker", "oversized speaker"),
        aliases=("speaker", "loa", "speaker"),
    ),
    # 52. Hộp đựng đồ
    "storage_box": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the storage box resting on an existing shelf or on the existing floor in picture 1."
        ),
        rule="Storage Box Placement: Must rest flat on an existing shelf or floor.",
        negative_terms=("floating storage box", "extra table"),
        aliases=("storage_box", "hộp đựng đồ", "hop dung do", "storage box"),
    ),
    # 53. Khay/kệ đựng đồ
    "storage_rack": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the storage organizer rack resting on an existing counter, shelf, or existing floor in picture 1."
        ),
        rule="Storage Rack Placement: Must sit flat and stable on an existing surface or floor.",
        negative_terms=("floating rack", "extra table"),
        aliases=("storage_rack", "khay/kệ đựng đồ", "khay dung do", "ke dung do", "storage rack"),
    ),
    # 54. Thùng rác
    "trash_bin": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the waste trash bin standing upright on the existing floor beside furniture or in a corner in picture 1."
        ),
        rule="Trash Bin Placement: Must stand upright flat on the existing floor.",
        negative_terms=("floating trash bin", "trash bin on table"),
        aliases=("trash_bin", "thùng rác", "thung rac", "trash bin"),
    ),
    # 55. Máy tỉa lông
    "trimmer": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the compact trimmer resting on an existing vanity or shelf in picture 1. "
            "Realistic small handheld scale (~15cm). Do NOT add any new furniture."
        ),
        rule="Trimmer Placement: Must rest on an existing vanity or shelf.",
        negative_terms=("floating trimmer", "extra table", "giant trimmer", "oversized trimmer"),
        aliases=("trimmer", "máy tỉa lông", "may tia long"),
    ),
    # 56. TV
    "tv": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the television standing directly on top of the existing TV console shelf or cabinet in picture 1, "
            "feet flat with zero gap. The television MUST NOT be mounted or hung on the wall, If mounting it on the wall, remove the TV stand.; stand feet must touch shelf. Do NOT add new furniture."
        ),
        rule=(
            "Television Placement: Must rest flat on top of existing TV console shelf/cabinet with zero gap under feet. "
            "Do not hang on wall or add new furniture."
            "Mount the tivi lower than the air conditioner"
        ),
        negative_terms=(
            "wall mounted TV with legs",
            "TV with stand hanging on wall",
            "TV floating in air",
            "gap under TV feet",
            "hovering TV",
            "TV floating above cabinet",
            "extra TV cabinet",
            "new TV console",
        ),
        aliases=("tv", "tivi", "television"),
    ),
    # 57. Máy hút bụi
    "vacuum_cleaner": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the vacuum cleaner standing or resting on the existing floor in picture 1."
        ),
        rule="Vacuum Cleaner Placement: Must rest flat on the existing floor.",
        negative_terms=("floating vacuum cleaner", "vacuum cleaner on table"),
        aliases=("vacuum_cleaner", "máy hút bụi", "may hut bui", "vacuum cleaner"),
    ),
    # 58. Máy giặt
    "washing_machine": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the washing machine standing upright firmly on the existing floor in picture 1. "
            "Do NOT place on tables or counters."
        ),
        rule="Washing Machine Placement: Must stand upright with its base flat on the existing floor, never floating or on furniture.",
        negative_terms=("floating washing machine", "washing machine on table"),
        aliases=("washing_machine", "máy giặt", "may giat", "washer"),
    ),
    # 59. Cây nước nóng lạnh
    "water_dispenser": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place slim water dispenser standing upright flat on the floor against the wall in picture 1. "
            "Scale to counter height (~90cm-1m tall). Do NOT place in walkway or foreground."
        ),
        rule="Water Dispenser Placement: Stand upright flat on existing floor against wall. Do NOT block walkway or foreground.",
        negative_terms=(
            "water purifier in walkway",
            "water purifier in foreground",
            "water purifier in front of island",
            "water purifier taller than counter",
            "giant water dispenser",
            "oversized water dispenser",
            "floating water dispenser",
            "water dispenser on table",
            "blocking walkway",
        ),
        aliases=("water_dispenser", "cây nước nóng lạnh", "cay nuoc nong lanh", "water dispenser"),
    ),
    # 60. Máy tăm nước
    "water_flosser": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place the compact water flosser standing upright on an existing vanity counter or shelf in picture 1. "
            "Realistic small handheld scale (~20-25cm tall)."
        ),
        rule="Water Flosser Placement: Must stand upright on an existing bathroom counter or shelf.",
        negative_terms=("floating water flosser", "giant water flosser", "oversized water flosser"),
        aliases=("water_flosser", "máy tăm nước", "may tam nuoc"),
    ),
    # 61. Máy nước nóng
    "water_heater": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Mount securely onto the existing bathroom wall plane at realistic installation height in picture 1. "
            "Do NOT modify the wall or place on the floor."
        ),
        rule="Water Heater Placement: Must be mounted on the existing bathroom wall plane, never placed on the floor.",
        negative_terms=("water heater on floor", "floating water heater"),
        aliases=("water_heater", "máy nước nóng", "may nuoc nong", "water heater"),
    ),
    # 62. Máy lọc nước
    "water_purifier": CategoryRuleConfig(
        directive_template=(
            "- Picture {index} ({pname}): Place slim floor-standing water purifier standing upright flat on the floor against the wall in picture 1. "
            "Scale to counter height (~90cm-1m tall). Do NOT place in walkway or foreground."
        ),
        rule="Water Purifier Placement: Stand upright on floor against wall or corner. Do NOT block walkway or foreground.",
        negative_terms=(
            "water purifier in walkway",
            "water purifier in foreground",
            "water purifier in front of island",
            "water purifier taller than counter",
            "giant water purifier",
            "oversized water purifier",
            "floating water purifier",
            "extra table",
        ),
        aliases=("water_purifier", "máy lọc nước", "may loc nuoc", "water purifier"),
    ),
}


class ImageEditPromptBuilder:
    """Class chuyên biệt xây dựng prompt và negative prompt theo từng sản phẩm cho Qwen Image Edit."""

    MAX_PROMPT_WORDS: int = 2000
    MAX_PROMPT_CHARS: int = 4000
    MAX_NEGATIVE_PROMPT_CHARS: int = 2500

    UNIVERSAL_RULES: tuple[str, ...] = (
        "Strictly Single Instance & Exact Count: Place EXACTLY ONE instance of each referenced product.",
        "Room Structure Preservation: Keep original room 100% unchanged (walls, floor, ceiling, furniture). Do NOT modify or carve into existing cabinets.",
        "Zero Invented Furniture or Items: Do NOT add new tables, desks, cabinets, dish racks, or decor. Place only on existing surfaces.",
        "Realistic Scale: Match real-world physical scale and room depth. Countertop items stay compact. Never magnify.",
        "Clear Foreground: Foreground floor must stay empty. Do not place items in the immediate foreground near camera.",
    )

    BASE_NEGATIVE_TERMS: tuple[str, ...] = (
        "floating objects",
        "floating in mid-air",
        "floating above surface",
        "unanchored object",
        "changing room",
        "altering room structure",
        "altering walls",
        "different floor",
        "different ceiling",
        "extra furniture",
        "extra tables",
        "extra desks",
        "extra chairs",
        "wooden table in foreground",
        "new furniture",
        "invented objects",
        "unrequested decorations",
        "duplicate products",
        "duplicate items",
        "duplicate appliances",
        "duplicate rice cooker",
        "multiple rice cookers",
        "two rice cookers",
        "three rice cookers",
        "cloned products",
        "twin products",
        "repeated products",
        "repeated appliances",
        "multiple copies of same product",
        "two of the same item",
        "multiple appliances",
        "carved into cabinet",
        "cut into cabinets",
        "embedded in counter",
        "dish rack on wall",
        "extra boxes",
        "pedestals",
        "stands under products",
        "black box on floor",
        "cut into cabinets",
        "embedded in counter",
        "misaligned with wall",
        "misaligned with floor",
        "wrong surface",
        "in foreground",
        "near camera",
        "blocking foreground",
        "blocking walkway",
        "middle of room",
        "oversized products",
        "huge objects",
        "wrong scale",
        "distorted scale",
        "distorted products",
        "text overlay",
        "watermark",
        "promotional banner",
        "low quality",
        "blurry",
    )

    def __init__(self) -> None:
        pass

    def build_product_spec(
        self, product: Mapping[str, Any], index: int
    ) -> ProductPlacementSpec:
        """Xây dựng đặc tả (directive, rule, negative_terms) riêng cho từng sản phẩm."""
        group = (product.get("group_code") or "").strip().lower()
        name = (product.get("product_name") or "").strip().lower()
        pname = product.get("product_name") or "Product"
        placement = product.get("placement") or "on the existing floor or shelf naturally"

        # 1. Tra cứu trực tiếp theo group_code trong CATEGORY_RULES
        cfg = CATEGORY_RULES.get(group)

        # 2. Nếu chưa thấy, tra cứu theo aliases trong group_code hoặc product_name
        if cfg is None:
            for _code, cat_cfg in CATEGORY_RULES.items():
                if any(alias in group or alias in name for alias in cat_cfg.aliases):
                    cfg = cat_cfg
                    break

        # 3. Nếu tìm thấy cấu hình quy tắc
        if cfg is not None:
            directive = cfg.directive_template.format(index=index, pname=pname)
            return ProductPlacementSpec(
                directive=directive,
                rule=cfg.rule,
                negative_terms=cfg.negative_terms,
            )

        # 4. Dự phòng cho trường hợp sản phẩm hoàn toàn nằm ngoài 62 nhóm
        directive = (
            f"- Picture {index} ({pname}): Place on an existing surface or existing floor in picture 1 {placement}. "
            "Do NOT add any new tables, desks, or furniture."
        )
        return ProductPlacementSpec(directive=directive, rule=None)

    def build_prompt(
        self,
        products: list[dict[str, Any]],
        max_products: int = 2,
        room_type: str | None = None,
    ) -> str:
        """Xây dựng toàn bộ prompt từ danh sách sản phẩm thực tế:
        - Mở đầu bằng lệnh chuẩn mực: Mount product in picture 2, picture 3 to picture 1. Do not change room structure.
        - Chỉ lấy những vật thể nằm trong danh sách gợi ý.
        - Giữ nguyên cấu trúc phòng 100%, cấm thêm đồ đạc ngoài danh sách.
        """
        selected = products[:max_products]
        if not selected:
            return self.default_prompt

        num_products = len(selected)
        has_tv = any(
            (p.get("group_code") == "tv")
            or ("tv" in (p.get("product_name") or "").lower().split())
            or ("tivi" in (p.get("product_name") or "").lower())
            or ("television" in (p.get("product_name") or "").lower())
            for p in selected
        )
        tv_surface_note = "; TV flat on console shelf" if has_tv else ""
        scene_analysis = ROOM_SCENE_ANALYSIS.get((room_type or "").strip().lower())

        # Giữ nguyên header phòng khách/các phòng chưa có quy tắc riêng.
        if scene_analysis:
            surface_step = 2
            anchor_step = 3
            instance_step = 4
            analysis_block = f"1. {scene_analysis}\n"
        else:
            surface_step = 1
            anchor_step = 2
            instance_step = 3
            analysis_block = ""

        if num_products == 1:
            header = (
                "Mount product in picture 2 to picture 1. Do not change room structure.\n"
                f"{analysis_block}"
                f"{surface_step}. Surface Detection: Identify walls, ceiling, floor, counters, shelves in picture 1.\n"
                f"{anchor_step}. Anchored Placement: Mount product flat on matching surface (wall units high near ceiling; countertop items on counters; floor units on floor{tv_surface_note}). Never float.\n"
                f"{instance_step}. Single Instance: Place exactly one item. Strictly NO duplicate or cloned items."
            )
        else:
            refs = " and ".join(f"picture {i}" for i in range(2, num_products + 2))
            header = (
                f"Mount products in {refs} to picture 1. Do not change room structure.\n"
                f"{analysis_block}"
                f"{surface_step}. Surface Detection: Identify walls, ceiling, floor, counters, shelves in picture 1.\n"
                f"{anchor_step}. Anchored Placement: Mount each product flat on matching surface (wall units high near ceiling; countertop items on counters; floor units on floor{tv_surface_note}). Never float.\n"
                f"{instance_step}. Single Instance: Place exactly one instance of each product. Strictly NO duplicate or twin items."
            )

        specs: list[ProductPlacementSpec] = [
            self.build_product_spec(prod, idx)
            for idx, prod in enumerate(selected, start=2)
        ]

        # Ghép các chỉ dẫn cho từng Picture
        items_block = "\n".join(spec.directive for spec in specs)

        # Ghép các quy tắc cụ thể cho từng sản phẩm (loại bỏ trùng lặp nếu cả 2 cùng loại)
        rules: list[str] = []
        rule_num = 1
        seen_rules: set[str] = set()

        for spec in specs:
            if spec.rule and spec.rule not in seen_rules:
                seen_rules.add(spec.rule)
                rules.append(f"{rule_num}. {spec.rule}")
                rule_num += 1

        # Thêm các quy tắc bất biến về bảo tồn phòng và cấm đồ vật phát sinh
        for u_rule in self.UNIVERSAL_RULES:
            rules.append(f"{rule_num}. {u_rule}")
            rule_num += 1

        rules_block = "\n".join(rules)
        full_prompt = (
            f"{header}\n\n"
            f"{items_block}\n\n"
            f"**RULES & PRESERVATION**\n"
            f"{rules_block}"
        )
        words = full_prompt.split()
        if len(words) > self.MAX_PROMPT_WORDS:
            full_prompt = " ".join(words[:self.MAX_PROMPT_WORDS]).rstrip()
        if len(full_prompt) > self.MAX_PROMPT_CHARS:
            full_prompt = full_prompt[:self.MAX_PROMPT_CHARS].rstrip()
        return full_prompt

    def build_negative_prompt(
        self, products: list[dict[str, Any]], max_products: int = 2
    ) -> str:
        """Xây dựng negative prompt linh hoạt theo từng sản phẩm:
        - Đưa vào các điều kiện cấm thay đổi phòng, cấm sinh bàn ghế mới và các negative terms riêng cho từng sản phẩm.
        """
        selected = products[:max_products]
        specific_neg: list[str] = []
        seen: set[str] = set()

        # 1. Các terms riêng theo sản phẩm
        for idx, prod in enumerate(selected, start=2):
            spec = self.build_product_spec(prod, idx)
            for term in spec.negative_terms:
                if term not in seen:
                    seen.add(term)
                    specific_neg.append(term)

        # 2. Các terms chung chống thay đổi phòng và chống sinh đồ mới
        for term in self.BASE_NEGATIVE_TERMS:
            if term not in seen:
                seen.add(term)
                specific_neg.append(term)

        full_neg = ", ".join(specific_neg)
        if len(full_neg) > self.MAX_NEGATIVE_PROMPT_CHARS:
            truncated = full_neg[:self.MAX_NEGATIVE_PROMPT_CHARS]
            if "," in truncated:
                full_neg = truncated.rsplit(",", 1)[0].rstrip()
            else:
                full_neg = truncated.rstrip()
        return full_neg

    @property
    def default_prompt(self) -> str:
        """Prompt mặc định dự phòng tổng quát khi danh sách sản phẩm rỗng."""
        u_rules = "\n".join(f"{i+2}. {rule}" for i, rule in enumerate(self.UNIVERSAL_RULES))
        prompt = (
            "Mount products in picture 2 and picture 3 to picture 1. Do not change room structure.\n"
            "1. Surface Detection: Identify walls, ceiling, floor, counters, shelves in picture 1.\n"
            "2. Anchored Placement: Place each product on matching surface (wall units high near ceiling; tabletop on counters; floor units on floor). Never float.\n"
            "3. Single Instance: Place exactly one instance of each product. Strictly NO duplicate or twin items.\n\n"
            "- Picture 2, Picture 3: Product reference images to place inside picture 1.\n\n"
            "**RULES & PRESERVATION**\n"
            "1. Product Placement: Place each product according to realistic function. Never float. Do NOT add new tables.\n"
            f"{u_rules}"
        )
        words = prompt.split()
        if len(words) > self.MAX_PROMPT_WORDS:
            prompt = " ".join(words[:self.MAX_PROMPT_WORDS]).rstrip()
        if len(prompt) > self.MAX_PROMPT_CHARS:
            prompt = prompt[:self.MAX_PROMPT_CHARS].rstrip()
        return prompt

    @property
    def default_negative_prompt(self) -> str:
        """Negative prompt mặc định tổng quát."""
        # Đồng nhất với `build_negative_prompt`: chỉ giữ mỗi negative term một lần.
        neg = ", ".join(dict.fromkeys(self.BASE_NEGATIVE_TERMS))
        if len(neg) > self.MAX_NEGATIVE_PROMPT_CHARS:
            truncated = neg[:self.MAX_NEGATIVE_PROMPT_CHARS]
            if "," in truncated:
                neg = truncated.rsplit(",", 1)[0].rstrip()
            else:
                neg = truncated.rstrip()
        return neg


# Instance mặc định dùng chung
_DEFAULT_BUILDER = ImageEditPromptBuilder()


def build_room_edit_prompt(
    products: list[dict[str, Any]], max_products: int = 2, room_type: str | None = None
) -> str:
    """Hàm tiện ích tương thích ngược để xây dựng prompt."""
    return _DEFAULT_BUILDER.build_prompt(
        products, max_products=max_products, room_type=room_type
    )


def build_room_edit_negative_prompt(
    products: list[dict[str, Any]], max_products: int = 2
) -> str:
    """Hàm tiện ích tương thích ngược để xây dựng negative prompt."""
    return _DEFAULT_BUILDER.build_negative_prompt(products, max_products=max_products)


DEFAULT_ROOM_EDIT_PROMPT = _DEFAULT_BUILDER.default_prompt
DEFAULT_ROOM_EDIT_NEGATIVE_PROMPT = _DEFAULT_BUILDER.default_negative_prompt
