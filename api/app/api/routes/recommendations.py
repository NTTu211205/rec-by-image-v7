"""Route gợi ý: ảnh -> detect (Cell 10.13) -> recommend (Cell 10.10)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from app.core.config import PRODUCTS_PER_GROUP_MAX, TOP_K_MAX, Settings, get_settings
from app.core.exceptions import BusyError, ModelNotReadyError, NotFoundError
from app.dependencies import AppStateDep
from app.api.openapi import ERROR_CONTENT
from app.schemas.recommendation import RecommendationResponse
from app.services import group_scores
from app.services.images import load_image_bytes
from app.services.qwen_image_edit import QwenImageEditClient

logger = logging.getLogger("app.routes.recommendations")
router = APIRouter(prefix="/api/v1", tags=["recommendations"])


def _validate_room(room_type: str | None, state) -> str:
    allowed = set(state.recommender.room_types)
    room = room_type or state.settings.default_room_type
    if room not in allowed:
        raise NotFoundError(
            f"room_type '{room}' không hợp lệ. Chọn một trong: {', '.join(sorted(allowed))}."
        )
    return room


def _validate_category(category_id: str | None) -> str | None:
    if category_id is None:
        return None
    if category_id not in group_scores.CATEGORY_NAMES:
        raise NotFoundError(
            f"category_id '{category_id}' không hợp lệ. Chọn một trong: "
            f"{', '.join(sorted(group_scores.CATEGORY_NAMES))}."
        )
    return category_id


def _price_range(min_price: float | None, max_price: float | None):
    if min_price is None and max_price is None:
        return None
    return (0.0 if min_price is None else min_price, float("inf") if max_price is None else max_price)


@router.post(
    "/recommendations/image",
    response_model=RecommendationResponse,
    summary="Upload ảnh phòng → danh sách nhóm sản phẩm gợi ý",
    description="Upload ảnh phòng (JPEG/PNG/WEBP) → detect → xếp hạng nhóm sản phẩm.",
    response_description="Các nhóm sản phẩm đã xếp hạng",
    operation_id="recommendFromImage",
    responses={
        200: {"description": "Gợi ý thành công."},
        400: {"description": "Ảnh không hợp lệ.", "content": ERROR_CONTENT},
        404: {"description": "Tham số ngoài miền hợp lệ.", "content": ERROR_CONTENT},
        429: {"description": "Quá nhiều yêu cầu đang xử lý.", "content": ERROR_CONTENT},
        503: {"description": "Model chưa sẵn sàng.", "content": ERROR_CONTENT},
    },
)
async def recommend_from_image(
    state: AppStateDep,
    settings: Settings = Depends(get_settings),
    image: UploadFile = File(
        ...,
        description="Ảnh phòng: JPEG/PNG/WEBP.",
    ),
    room_type: str | None = Query(
        None,
        description=(
            "Mã phòng; bỏ trống → detect tất cả nhãn Grounding DINO và dùng "
            "`DEFAULT_ROOM_TYPE` để xếp hạng gợi ý."
        ),
        examples=["living_room"],
    ),
    room_type_form: str | None = Form(
        None,
        alias="room_type",
        description="Mã phòng gửi qua form-data.",
    ),
    top_k: int | None = Query(
        None, ge=1, le=TOP_K_MAX,
        description=f"Số nhóm trả về (1–{TOP_K_MAX}).",
        examples=[5],
    ),
    min_score: float | None = Query(
        None, ge=0,
        description="Ngưỡng `final_score` tối thiểu.",
        examples=[0.05],
    ),
    exclude_detected: bool | None = Query(
        None,
        description="Loại nhóm đã có trong ảnh.",
    ),
    products_per_group: int | None = Query(
        None, ge=1, le=PRODUCTS_PER_GROUP_MAX,
        description=f"Số sản phẩm mỗi nhóm (1–{PRODUCTS_PER_GROUP_MAX}).",
        examples=[3],
    ),
    min_price: float | None = Query(None, ge=0, description="Giá thấp nhất."),
    max_price: float | None = Query(None, ge=0, description="Giá cao nhất."),
    category_id: str | None = Query(
        None,
        description="Chỉ gợi ý trong một category.",
        examples=["electronics"],
    ),
    confidence: float | None = Query(
        None, ge=0, le=1,
        description="Ngưỡng confidence của detector.",
        examples=[0.5],
    ),
    generate_image: bool | None = Query(
        None,
        description="Gọi Qwen Gateway ghép sản phẩm vào ảnh phòng qua endpoint /v1/images/edits.",
        examples=[True],
    ),
    edit_prompt: str | None = Query(
        None,
        description="Câu prompt tuỳ biến cho Qwen edit (mặc định: Photorealistic Interior Product Placement Prompt).",
        examples=["Mount product in picture 2, picture 3 to picture 1. Not changes room structure"],
    ),
) -> RecommendationResponse:
    if not state.ready:
        raise ModelNotReadyError("Model chưa sẵn sàng.")

    requested_room = room_type or room_type_form
    # Không có room_type: detector chạy toàn bộ nhãn, rồi suy luận phòng từ
    # các vật thể phát hiện được trước khi xếp hạng sản phẩm.
    detection_room = requested_room or None
    provided_room = _validate_room(requested_room, state) if requested_room else None
    category = _validate_category(category_id)
    if top_k is not None and not (1 <= top_k <= TOP_K_MAX):
        raise NotFoundError(f"top_k phải trong [1, {TOP_K_MAX}].")
    if products_per_group is not None and not (1 <= products_per_group <= PRODUCTS_PER_GROUP_MAX):
        raise NotFoundError(f"products_per_group phải trong [1, {PRODUCTS_PER_GROUP_MAX}].")

    data = await image.read()
    if len(data) > settings.max_upload_bytes:
        raise NotFoundError(f"Ảnh vượt {settings.max_upload_mb:g} MB.")
    img = load_image_bytes(
        data, max_pixels=settings.max_image_pixels, min_side=settings.min_image_side
    )

    if not state.try_acquire():
        raise BusyError("Hệ thống đang xử lý quá nhiều yêu cầu, thử lại sau.")
    try:
        detected = await asyncio.get_running_loop().run_in_executor(
            state.executor,
            state.detector.detect,
            img,
            confidence if confidence is not None else settings.model_confidence,
            detection_room,
        )
        if provided_room is None:
            room, room_detection_scores = state.recommender.infer_room_type(detected)
            room_type_source = "inferred" if any(room_detection_scores.values()) else "default"
        else:
            room = provided_room
            room_detection_scores = None
            room_type_source = "provided"

        result = state.recommender.recommend(
            detected,
            room_type=room,
            top_k=top_k,
            min_score=min_score,
            exclude_detected=(
                settings.default_exclude_detected if exclude_detected is None else exclude_detected
            ),
            products_per_group=products_per_group,
            price_range=_price_range(min_price, max_price),
            category_id=category,
        )
        result["room_type_source"] = room_type_source
        result["room_detection_scores"] = room_detection_scores
    finally:
        state.release()

    # Ghép sản phẩm vào phòng bằng Qwen-Image-Edit gateway qua endpoint /v1/images/edits
    should_generate = (
        generate_image if generate_image is not None else settings.auto_generate_image
    )
    if should_generate:
        recs = result.get("recommendations", [])
        candidate_products: list[dict[str, Any]] = []

        # Lấy sản phẩm đầu tiên ở 2 nhóm cao nhất để tạo 1 ảnh duy nhất ở cuối cùng
        for r in recs[:2]:
            products = r.get("products", [])
            if products:
                p = products[0]
                if p.get("image_url"):
                    candidate_products.append({
                        "product_id": p.get("product_id"),
                        "product_name": p.get("product_name"),
                        "image_url": p.get("image_url"),
                        "group_code": r.get("group_code"),
                        "category_id": r.get("category_id"),
                    })

        if candidate_products:
            client = QwenImageEditClient(
                base_url=settings.qwen_gateway_url,
                api_key=settings.qwen_api_key,
                timeout=settings.qwen_gateway_timeout_s,
                connect_timeout=settings.qwen_gateway_connect_timeout_s,
            )
            edit_res = await client.mount_products_to_room(
                room_image=img,
                products=candidate_products,
                room_type=room,
                custom_prompt=edit_prompt,
            )
            result["generation_prompt"] = edit_res.prompt
            if edit_res.success and edit_res.generated_image:
                result["generated_image"] = edit_res.generated_image
                result["generation_status"] = "success"
            else:
                result["generation_status"] = "failed"
                result["generation_error"] = edit_res.error
        else:
            result["generation_status"] = "skipped"
            result["generation_error"] = (
                "Không có sản phẩm trong 2 nhóm đầu hoặc sản phẩm thiếu image_url."
            )
    else:
        result["generation_status"] = "skipped"

    logger.info(
        "recommend done",
        extra={"event": "recommend", "detected": len(detected),
               "groups": len(result["recommendations"]), "room_type": room},
    )
    return RecommendationResponse(**result)
