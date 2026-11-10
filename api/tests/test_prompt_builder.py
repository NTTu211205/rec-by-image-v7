"""Unit test cho ImageEditPromptBuilder."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.prompt_builder import (
    CATEGORY_RULES,
    DEFAULT_ROOM_EDIT_NEGATIVE_PROMPT,
    DEFAULT_ROOM_EDIT_PROMPT,
    ImageEditPromptBuilder,
    ProductPlacementSpec,
    build_room_edit_negative_prompt,
    build_room_edit_prompt,
)


def test_builder_default_prompt() -> None:
    builder = ImageEditPromptBuilder()
    prompt = builder.default_prompt
    assert "Mount products in picture 2 and picture 3 to picture 1" in prompt
    assert "Do not change room structure" in prompt
    assert "Zero Invented Furniture or Items" in prompt
    assert "Room Structure & Background Preservation" in prompt
    # Default prompt không được chứa tivi hay máy lạnh
    assert "television" not in prompt.lower()
    assert "air conditioner" not in prompt.lower()
    assert prompt == DEFAULT_ROOM_EDIT_PROMPT


def test_builder_default_negative_prompt() -> None:
    builder = ImageEditPromptBuilder()
    neg = builder.default_negative_prompt
    assert "changing room" in neg
    assert "altering room structure" in neg
    assert "extra furniture" in neg
    assert "extra tables" in neg
    assert "wooden table in foreground" in neg
    assert "floating objects" in neg
    assert "tv" not in neg.lower()
    assert neg == DEFAULT_ROOM_EDIT_NEGATIVE_PROMPT


def test_builder_spec_for_tv() -> None:
    builder = ImageEditPromptBuilder()
    spec = builder.build_product_spec({"group_code": "tv", "product_name": "Samsung TV"}, index=2)
    assert isinstance(spec, ProductPlacementSpec)
    assert "Picture 2 (Samsung TV)" in spec.directive
    assert "existing TV console shelf or cabinet" in spec.directive
    assert "MUST NOT be mounted or hung on the wall" in spec.directive
    assert spec.rule is not None
    assert "Television Placement:" in spec.rule
    assert "wall mounted TV with legs" in spec.negative_terms


def test_builder_spec_for_ac() -> None:
    builder = ImageEditPromptBuilder()
    spec = builder.build_product_spec({"group_code": "air_conditioner", "product_name": "Livotec AC"}, index=3)
    assert "Picture 3 (Livotec AC)" in spec.directive
    assert "existing upper wall plane near the ceiling" in spec.directive
    assert spec.rule is not None
    assert "Air Conditioner Placement:" in spec.rule


def test_builder_spec_for_ceiling_fan() -> None:
    builder = ImageEditPromptBuilder()
    spec = builder.build_product_spec({"group_code": "ceiling_fan", "product_name": "Quạt trần Panasonic"}, index=2)
    assert "existing ceiling" in spec.directive.lower()
    assert spec.rule is not None
    assert "Ceiling Fan Placement:" in spec.rule


def test_builder_spec_for_gas_stove() -> None:
    builder = ImageEditPromptBuilder()
    spec = builder.build_product_spec({"group_code": "gas_stove", "product_name": "Bếp gas Sunhouse"}, index=2)
    assert "existing kitchen counter" in spec.directive
    assert "Do NOT add or invent any new tables" in spec.directive
    assert spec.rule is not None
    assert "Gas Stove Placement:" in spec.rule
    assert "gas stove on floor" in spec.negative_terms


def test_builder_spec_for_induction_and_infrared_cooker() -> None:
    builder = ImageEditPromptBuilder()
    # Bếp từ
    spec_ind = builder.build_product_spec({"group_code": "induction_cooker", "product_name": "Bếp từ Kangaroo"}, index=2)
    assert "existing counter, table, or shelf" in spec_ind.directive
    assert "Do NOT add any new tables" in spec_ind.directive
    assert spec_ind.rule is not None
    assert "Induction Cooker Placement:" in spec_ind.rule
    assert "induction cooker on floor" in spec_ind.negative_terms

    # Bếp hồng ngoại
    spec_inf = builder.build_product_spec({"group_code": "infrared_cooker", "product_name": "Bếp hồng ngoại Sunhouse"}, index=3)
    assert "existing counter or table" in spec_inf.directive
    assert "Do NOT add any new tables" in spec_inf.directive
    assert spec_inf.rule is not None
    assert "Infrared Cooker Placement:" in spec_inf.rule
    assert "infrared cooker on floor" in spec_inf.negative_terms


def test_builder_spec_for_refrigerator() -> None:
    builder = ImageEditPromptBuilder()
    spec = builder.build_product_spec(
        {"group_code": "refrigerator", "product_name": "Tủ lạnh Toshiba Inverter 236L"},
        index=2,
    )
    assert "existing floor against the wall" in spec.directive
    assert "Do NOT place on tables or counters" in spec.directive
    assert spec.rule is not None
    assert "Refrigerator Placement:" in spec.rule
    assert "floating refrigerator" in spec.negative_terms
    assert "refrigerator on table" in spec.negative_terms


def test_builder_spec_for_oven() -> None:
    builder = ImageEditPromptBuilder()
    spec = builder.build_product_spec(
        {"group_code": "oven", "product_name": "Lò nướng Sharp 32L"},
        index=2,
    )
    assert "existing counter or existing shelf" in spec.directive
    assert "Do NOT add any new tables or cabinets" in spec.directive
    assert spec.rule is not None
    assert "Oven Placement:" in spec.rule
    assert "oven on floor" in spec.negative_terms


def test_builder_spec_for_all_62_categories() -> None:
    """Đảm bảo mọi nhóm trong mwg.categories (62 nhóm) đều có rule và directive chuẩn xác."""
    builder = ImageEditPromptBuilder()
    assert len(CATEGORY_RULES) == 62

    matrix_file = Path(__file__).resolve().parent.parent / "artifacts" / "matrix" / "category_room_scores.json"
    if matrix_file.is_file():
        scores = json.loads(matrix_file.read_text(encoding="utf-8"))
        assert len(scores) == 62

    for cat_code in CATEGORY_RULES:
        spec = builder.build_product_spec(
            {"group_code": cat_code, "product_name": f"Product {cat_code}"},
            index=2,
        )
        assert f"Picture 2 (Product {cat_code})" in spec.directive
        assert "picture 1" in spec.directive
        assert spec.rule is not None
        assert len(spec.rule) > 10


def test_builder_spec_for_unknown_fallback() -> None:
    builder = ImageEditPromptBuilder()
    spec = builder.build_product_spec(
        {"group_code": "unknown_item", "product_name": "Vật thể lạ", "placement": "in the corner"},
        index=2,
    )
    assert "Picture 2 (Vật thể lạ)" in spec.directive
    assert "Do NOT add any new tables, desks, or furniture" in spec.directive
    assert spec.rule is None
    assert spec.negative_terms == ()


def test_builder_prompt_without_tv() -> None:
    builder = ImageEditPromptBuilder()
    products = [
        {"group_code": "fan", "product_name": "Quạt đứng Senko", "placement": "on the floor"},
        {"group_code": "air_purifier", "product_name": "Máy lọc Sharp", "placement": "on the floor near the corner"},
    ]
    prompt = builder.build_prompt(products)
    prompt_lower = prompt.lower()
    assert "television" not in prompt_lower
    assert "tivi" not in prompt_lower
    assert " tv " not in prompt_lower
    assert "Quạt đứng Senko" in prompt
    assert "Máy lọc Sharp" in prompt

    neg = builder.build_negative_prompt(products)
    assert "tv" not in neg.lower()


def test_builder_prompt_with_tv() -> None:
    builder = ImageEditPromptBuilder()
    products = [
        {"group_code": "tv", "product_name": "LG OLED TV", "placement": "on the TV stand"},
        {"group_code": "air_conditioner", "product_name": "Daikin Inverter", "placement": "high on wall"},
    ]
    prompt = builder.build_prompt(products)
    assert "Television Placement:" in prompt
    assert "Air Conditioner Placement:" in prompt
    assert "LG OLED TV" in prompt
    assert "Daikin Inverter" in prompt

    neg = builder.build_negative_prompt(products)
    assert "wall mounted tv with legs" in neg.lower()


def test_builder_prompt_with_refrigerator_and_stove() -> None:
    builder = ImageEditPromptBuilder()
    products = [
        {"group_code": "refrigerator", "product_name": "Tủ lạnh Samsung", "placement": "on the floor"},
        {"group_code": "gas_stove", "product_name": "Bếp gas Sunhouse", "placement": "on the counter"},
    ]
    prompt = builder.build_prompt(products)
    assert "Refrigerator Placement:" in prompt
    assert "Gas Stove Placement:" in prompt
    assert "Tủ lạnh Samsung" in prompt
    assert "Bếp gas Sunhouse" in prompt

    neg = builder.build_negative_prompt(products)
    assert "floating refrigerator" in neg.lower()
    assert "gas stove on floor" in neg.lower()
    assert "tv" not in neg.lower()


def test_builder_empty_products() -> None:
    builder = ImageEditPromptBuilder()
    assert builder.build_prompt([]) == builder.default_prompt
    assert builder.build_negative_prompt([]) == builder.default_negative_prompt


def test_convenience_functions() -> None:
    prods = [{"group_code": "microwave", "product_name": "Lò vi sóng Sharp", "placement": "on the counter"}]
    p1 = build_room_edit_prompt(prods)
    n1 = build_room_edit_negative_prompt(prods)
    assert "Lò vi sóng Sharp" in p1
    assert "Microwave Placement:" in p1
    assert "television" not in p1.lower()
    assert "tv" not in n1.lower()
