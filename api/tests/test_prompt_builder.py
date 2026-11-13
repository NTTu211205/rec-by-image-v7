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
    category_prompt_name,
)


def test_builder_default_prompt() -> None:
    builder = ImageEditPromptBuilder()
    prompt = builder.default_prompt
    assert "Mount products in picture 2 and picture 3 to picture 1" in prompt
    assert "Do not change room structure" in prompt
    assert "Zero Invented Furniture or Items" in prompt
    assert "Room Structure Preservation" in prompt
    assert "Strictly Single Instance" in prompt
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
    assert "duplicate products" in neg
    assert "twin products" in neg
    assert "tv" not in neg.lower()
    assert neg == DEFAULT_ROOM_EDIT_NEGATIVE_PROMPT


def test_builder_keeps_living_room_prompt_unchanged() -> None:
    builder = ImageEditPromptBuilder()
    products = [{"group_code": "speaker", "product_name": "Loa"}]

    assert builder.build_prompt(products) == builder.build_prompt(
        products, room_type="living_room"
    )


def test_builder_adds_kitchen_scene_analysis_before_placement() -> None:
    prompt = ImageEditPromptBuilder().build_prompt(
        [{"group_code": "rice_cooker", "product_name": "Nồi cơm điện"}],
        room_type="kitchen",
    )

    assert "Kitchen Scene Analysis" in prompt
    assert "kitchen cabinets" in prompt
    assert "countertops" in prompt
    assert prompt.index("Kitchen Scene Analysis") < prompt.index("Picture 2")


def test_builder_adds_bedroom_scene_analysis_before_placement() -> None:
    prompt = ImageEditPromptBuilder().build_prompt(
        [{"group_code": "lamp", "product_name": "Đèn ngủ"}],
        room_type="bedroom",
    )

    assert "Bedroom Scene Analysis" in prompt
    assert "existing bed" in prompt
    assert "mirror" in prompt
    assert prompt.index("Bedroom Scene Analysis") < prompt.index("Picture 2")


def test_builder_spec_for_tv() -> None:
    builder = ImageEditPromptBuilder()
    spec = builder.build_product_spec({"group_code": "tv", "product_name": "Samsung TV"}, index=2)
    assert isinstance(spec, ProductPlacementSpec)
    assert "Picture 2 (television)" in spec.directive
    assert "Samsung TV" not in spec.directive
    assert "existing TV console shelf or cabinet" in spec.directive
    assert "MUST NOT be mounted or hung on the wall" in spec.directive
    assert spec.rule is not None
    assert "Television Placement:" in spec.rule
    assert "wall mounted TV with legs" in spec.negative_terms


def test_builder_spec_for_ac() -> None:
    builder = ImageEditPromptBuilder()
    spec = builder.build_product_spec({"group_code": "air_conditioner", "product_name": "Livotec AC"}, index=3)
    assert "Picture 3 (air conditioner)" in spec.directive
    assert "Livotec AC" not in spec.directive
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
    assert "Copy exactly one visible refrigerator" in spec.directive
    assert "must be visible in the final image" in spec.directive
    assert "kitchen cabinet" in spec.directive.lower()
    assert "unobstructed empty floor area" in spec.directive
    assert "back against an uninterrupted wall" in spec.directive
    assert "least-obstructed outer image edge" in spec.directive
    assert "Do not overlap, cover, or stand in front" in spec.directive
    assert "Preserve every existing cabinet" in spec.directive
    assert spec.rule is not None
    assert "Refrigerator Placement:" in spec.rule
    assert "outer cabinet end" in spec.rule.lower()
    assert "floating appliance" in spec.negative_terms
    assert "appliance on table" in spec.negative_terms
    assert "appliance blocking walkway" in spec.negative_terms
    assert all("refrigerator" not in term for term in spec.negative_terms)


def test_builder_uses_a_compact_refrigerator_only_prompt() -> None:
    prompt = ImageEditPromptBuilder().build_prompt(
        [{"group_code": "refrigerator", "product_name": "Tủ lạnh Example 500L"}],
        room_type="kitchen",
    )

    assert "Mount the refrigerator in picture 2 into picture 1" in prompt
    assert "must visibly contain exactly one refrigerator" in prompt
    assert "unobstructed empty floor area" in prompt
    assert "least-obstructed outer image edge" in prompt
    assert "Never overlap, cover, or stand in front" in prompt
    assert "Tủ lạnh Example 500L" not in prompt
    assert "Kitchen Scene Analysis" not in prompt
    assert len(prompt.split()) < 150


def test_builder_uses_one_precise_prompt_for_refrigerator_and_rice_cooker() -> None:
    prompt = ImageEditPromptBuilder().build_prompt([
        {"group_code": "refrigerator", "product_name": "Ignored refrigerator name"},
        {"group_code": "rice_cooker", "product_name": "Ignored rice cooker name"},
    ], room_type="kitchen")

    assert "refrigerator in picture 2 and the rice cooker in picture 3" in prompt
    assert "exactly one refrigerator from picture 2" in prompt
    assert "exactly one compact rice cooker from picture 3" in prompt
    assert "HARD EXACT-COUNT CONSTRAINT" in prompt
    assert "Create no duplicate, clone, second unit" in prompt
    assert "First map every placement surface" in prompt
    assert "base cabinets, upper cabinets, each countertop" in prompt
    assert "unobstructed empty floor zone" in prompt
    assert "one existing clear kitchen countertop" in prompt
    assert "Ignored refrigerator name" not in prompt
    assert "Ignored rice cooker name" not in prompt


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
        assert f"Picture 2 ({category_prompt_name(cat_code)})" in spec.directive
        assert "picture 1" in spec.directive
        assert spec.rule is not None
        assert len(spec.rule) > 10


def test_builder_spec_for_unknown_fallback() -> None:
    builder = ImageEditPromptBuilder()
    spec = builder.build_product_spec(
        {"group_code": "unknown_item", "product_name": "Vật thể lạ", "placement": "in the corner"},
        index=2,
    )
    assert "Picture 2 (unknown item)" in spec.directive
    assert "Vật thể lạ" not in spec.directive
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
    assert "Picture 2 (fan)" in prompt
    assert "Picture 3 (air purifier)" in prompt
    assert "Quạt đứng Senko" not in prompt
    assert "Máy lọc Sharp" not in prompt

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
    assert "Picture 2 (television)" in prompt
    assert "Picture 3 (air conditioner)" in prompt
    assert "LG OLED TV" not in prompt
    assert "Daikin Inverter" not in prompt

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
    assert "Picture 2 (refrigerator)" in prompt
    assert "Picture 3 (gas stove)" in prompt
    assert "Tủ lạnh Samsung" not in prompt
    assert "Bếp gas Sunhouse" not in prompt

    neg = builder.build_negative_prompt(products)
    assert "floating appliance" in neg.lower()
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
    assert "Picture 2 (microwave)" in p1
    assert "Lò vi sóng Sharp" not in p1
    assert "Microwave Placement:" in p1
    assert "television" not in p1.lower()
    assert "tv" not in n1.lower()


def test_prompt_length_limit() -> None:
    builder = ImageEditPromptBuilder()
    assert len(builder.default_prompt.split()) < 2000
    assert len(builder.default_prompt) <= 4000

    # Test single product
    for code in CATEGORY_RULES:
        p = builder.build_prompt([{"group_code": code, "product_name": f"Product {code}"}])
        assert len(p.split()) < 2000, f"Exceeded 2000 words for {code}: {len(p.split())}"
        assert len(p) <= 4000, f"Exceeded 4000 chars for {code}: {len(p)}"

    # Test pair with longest text (kettle + tv, soy_milk_maker + tv)
    p_long = builder.build_prompt([
        {"group_code": "soy_milk_maker", "product_name": "Máy làm sữa hạt đa năng Tefal Ultra Blend High Speed"},
        {"group_code": "tv", "product_name": "Smart Tivi Neo QLED 8K 85 inch Samsung QA85QN900C"}
    ])
    assert len(p_long.split()) < 2000
    assert len(p_long) <= 4000
    assert len(p_long) > 0
