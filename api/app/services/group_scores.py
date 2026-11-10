"""Metadata nhóm sản phẩm: tên hiển thị và category cha.

File này chỉ chứa **nhãn**, không chứa điểm. Điểm phù hợp phòng W(c, r) đọc từ
`artifacts/matrix/category_room_scores.json` (xem `app/services/room_scores.py`);
catalog sản phẩm đọc từ PostgreSQL schema `mwg`.

Vì sao vẫn cần metadata trong code:

* `GROUP_TO_DETECTOR` (recommender) cần biết nhóm nào có detector proxy — đó là
  cấu hình thuật toán, không phải dữ liệu.
* `CATEGORY_NAMES` là 5 mục lớn (`electronics`…), nhãn tiếng Việt của chúng
  không nằm trong artifact nào.
* Tên nhóm trong file điểm là tên hiển thị (`"Máy điều hòa"`), nên cần ánh xạ
  tên -> `group_code` để ghép với `mwg.categories.category_name`.

62 nhóm dưới đây khớp 1-1 với `mwg.categories` và với
`artifacts/matrix/category_room_scores.json` — `scripts/verify.sh` và test
kiểm tra điều đó.
"""

from __future__ import annotations

# category_id -> tên hiển thị tiếng Việt.
CATEGORY_NAMES: dict[str, str] = {
    'appliance': 'Điện lạnh & Gia dụng',
    'electronics': 'Điện tử',
    'household': 'Gia đình',
    'kitchen_appliance': 'Nhà bếp',
    'personal_care': 'Chăm sóc cá nhân',
}

# group_code (trùng cột `subcategory` của CSV) -> tên hiển thị.
GROUP_NAMES: dict[str, str] = {
    'air_conditioner': 'Máy điều hòa',
    'air_cooler': 'Quạt điều hòa',
    'air_fryer': 'Nồi chiên không dầu',
    'air_purifier': 'Máy lọc không khí',
    'beverage_cooler': 'Tủ mát',
    'blender': 'Máy xay sinh tố',
    'body_scale': 'Cân sức khỏe',
    'ceiling_fan': 'Quạt trần',
    'clothes_dryer': 'Máy sấy quần áo',
    'coffee_maker': 'Máy pha cà phê',
    'dehumidifier': 'Máy hút ẩm',
    'desktop_pc': 'Máy tính để bàn',
    'dishwasher': 'Máy rửa chén',
    'electric_toothbrush': 'Bàn chải đánh răng điện',
    'facial_cleanser': 'Máy rửa mặt',
    'fan': 'Quạt điện',
    'foot_bath': 'Bồn ngâm chân',
    'freezer': 'Tủ đông',
    'game_console': 'Máy chơi game',
    'gas_stove': 'Bếp gas',
    'hair_dryer': 'Máy sấy tóc',
    'hair_styler': 'Máy tạo kiểu tóc',
    'heater_fan': 'Quạt sưởi',
    'indoor_camera': 'Camera trong nhà',
    'induction_cooker': 'Bếp từ',
    'infrared_cooker': 'Bếp hồng ngoại',
    'iron': 'Bàn ủi',
    'juicer': 'Máy ép trái cây',
    'karaoke_speaker': 'Loa karaoke',
    'kettle': 'Ấm đun nước',
    'lamp': 'Đèn điện/đèn sạc',
    'laptop': 'Laptop',
    'led_bulb': 'Bóng đèn LED',
    'massage_chair': 'Ghế massage',
    'massager': 'Thiết bị massage',
    'microphone': 'Microphone',
    'microwave': 'Lò vi sóng',
    'monitor': 'Màn hình máy tính',
    'mop_set': 'Bộ lau nhà',
    'outdoor_camera': 'Camera ngoài trời',
    'oven': 'Lò nướng',
    'power_strip': 'Ổ cắm điện',
    'projector': 'Máy chiếu',
    'range_hood': 'Máy hút mùi',
    'refrigerator': 'Tủ lạnh',
    'rice_cooker': 'Nồi cơm điện',
    'robot_vacuum': 'Robot hút bụi',
    'shaver': 'Máy cạo râu',
    'solar_light': 'Đèn năng lượng mặt trời',
    'soy_milk_maker': 'Máy làm sữa hạt',
    'speaker': 'Loa',
    'storage_box': 'Hộp đựng đồ',
    'storage_rack': 'Khay/kệ đựng đồ',
    'trash_bin': 'Thùng rác',
    'trimmer': 'Máy tỉa lông',
    'tv': 'TV',
    'vacuum_cleaner': 'Máy hút bụi',
    'washing_machine': 'Máy giặt',
    'water_dispenser': 'Cây nước nóng lạnh',
    'water_flosser': 'Máy tăm nước',
    'water_heater': 'Máy nước nóng',
    'water_purifier': 'Máy lọc nước',
}

# group_code -> category_id.
GROUP_CATEGORY: dict[str, str] = {
    'air_conditioner': 'appliance',
    'air_cooler': 'appliance',
    'air_fryer': 'kitchen_appliance',
    'air_purifier': 'appliance',
    'beverage_cooler': 'appliance',
    'blender': 'kitchen_appliance',
    'body_scale': 'personal_care',
    'ceiling_fan': 'appliance',
    'clothes_dryer': 'appliance',
    'coffee_maker': 'kitchen_appliance',
    'dehumidifier': 'appliance',
    'desktop_pc': 'electronics',
    'dishwasher': 'appliance',
    'electric_toothbrush': 'personal_care',
    'facial_cleanser': 'personal_care',
    'fan': 'appliance',
    'foot_bath': 'personal_care',
    'freezer': 'appliance',
    'game_console': 'electronics',
    'gas_stove': 'kitchen_appliance',
    'hair_dryer': 'household',
    'hair_styler': 'personal_care',
    'heater_fan': 'appliance',
    'indoor_camera': 'electronics',
    'induction_cooker': 'kitchen_appliance',
    'infrared_cooker': 'kitchen_appliance',
    'iron': 'household',
    'juicer': 'kitchen_appliance',
    'karaoke_speaker': 'electronics',
    'kettle': 'kitchen_appliance',
    'lamp': 'household',
    'laptop': 'electronics',
    'led_bulb': 'household',
    'massage_chair': 'personal_care',
    'massager': 'personal_care',
    'microphone': 'electronics',
    'microwave': 'kitchen_appliance',
    'monitor': 'electronics',
    'mop_set': 'household',
    'outdoor_camera': 'electronics',
    'oven': 'kitchen_appliance',
    'power_strip': 'household',
    'projector': 'electronics',
    'range_hood': 'kitchen_appliance',
    'refrigerator': 'appliance',
    'rice_cooker': 'kitchen_appliance',
    'robot_vacuum': 'household',
    'shaver': 'personal_care',
    'solar_light': 'household',
    'soy_milk_maker': 'kitchen_appliance',
    'speaker': 'electronics',
    'storage_box': 'household',
    'storage_rack': 'household',
    'trash_bin': 'household',
    'trimmer': 'personal_care',
    'tv': 'electronics',
    'vacuum_cleaner': 'household',
    'washing_machine': 'appliance',
    'water_dispenser': 'kitchen_appliance',
    'water_flosser': 'personal_care',
    'water_heater': 'appliance',
    'water_purifier': 'kitchen_appliance',
}


def group_name(group_code: str) -> str:
    """Tên hiển thị của nhóm; fallback về chính mã."""
    return GROUP_NAMES.get(group_code, group_code)


def category_name(category_id: str) -> str:
    """Tên hiển thị của category; fallback về chính mã."""
    return CATEGORY_NAMES.get(category_id, category_id)


# Tên hiển thị -> group_code, dùng để nối file điểm phòng (khoá theo tên) với
# catalog (khoá theo mã). Trùng tên là lỗi dữ liệu nên kiểm tra ngay khi import.
NAME_TO_GROUP_CODE: dict[str, str] = {name: code for code, name in GROUP_NAMES.items()}
if len(NAME_TO_GROUP_CODE) != len(GROUP_NAMES):  # pragma: no cover - lỗi dữ liệu tĩnh
    raise RuntimeError("GROUP_NAMES có tên hiển thị trùng — không ánh xạ được sang mã.")
