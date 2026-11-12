"""Route gợi ý: ảnh -> detect (Cell 10.13) -> recommend (Cell 10.10)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from app.core.config import PRODUCTS_PER_GROUP_MAX, TOP_K_MAX, Settings, get_settings
from app.core.exceptions import BusyError, ImageError, ModelNotReadyError, NotFoundError
from app.dependencies import AppStateDep
from app.api.openapi import ERROR_CONTENT
from app.schemas.recommendation import RecommendationResponse, ThreeRoomRecommendationResponse
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


def _price_range(
    min_price: float | None,
    max_price: float | None,
    default_min_price: float = 250_000.0,
) -> tuple[float, float]:
    lo = default_min_price if min_price is None else min_price
    hi = float("inf") if max_price is None else max_price
    return (lo, hi)


def _validate_ranking_parameters(
    top_k: int | None,
    products_per_group: int | None,
) -> None:
    if top_k is not None and not (1 <= top_k <= TOP_K_MAX):
        raise NotFoundError(f"top_k phải trong [1, {TOP_K_MAX}].")
    if products_per_group is not None and not (1 <= products_per_group <= PRODUCTS_PER_GROUP_MAX):
        raise NotFoundError(f"products_per_group phải trong [1, {PRODUCTS_PER_GROUP_MAX}].")


async def _load_upload_image(image: UploadFile, settings: Settings):
    """Đọc, kiểm tra kích thước và decode một ảnh multipart dùng chung cho các route."""
    data = await image.read()
    if len(data) > settings.max_upload_bytes:
        raise NotFoundError(f"Ảnh vượt {settings.max_upload_mb:g} MB.")
    return load_image_bytes(
        data, max_pixels=settings.max_image_pixels, min_side=settings.min_image_side
    )


def _candidate_products(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Lấy tối đa hai sản phẩm đầu bảng, đúng giới hạn 3 ảnh của images/edits."""
    candidates: list[dict[str, Any]] = []
    for recommendation in result.get("recommendations", [])[:2]:
        products = recommendation.get("products", [])
        if not products:
            continue
        product = products[0]
        if not product.get("image_url"):
            continue
        candidates.append({
            "product_id": product.get("product_id"),
            "product_name": product.get("product_name"),
            "image_url": product.get("image_url"),
            "group_code": recommendation.get("group_code"),
            "category_id": recommendation.get("category_id"),
            "group_category_id": recommendation.get("group_category_id"),
        })
    return candidates


async def _generate_recommendation_image(
    *,
    state: Any,
    settings: Settings,
    image: Any,
    recommendation: RecommendationResponse,
    edit_prompt: str | None = None,
) -> RecommendationResponse:
    """Gọi Qwen cho một kết quả, có semaphore dùng chung trên toàn API process."""
    result = recommendation.model_dump()
    candidates = _candidate_products(result)
    if not candidates:
        result["generation_status"] = "skipped"
        result["generation_error"] = (
            "Không có sản phẩm trong 2 nhóm đầu hoặc sản phẩm thiếu image_url."
        )
        return RecommendationResponse(**result)

    client = QwenImageEditClient(
        base_url=settings.qwen_gateway_url,
        api_key=settings.qwen_api_key,
        timeout=settings.qwen_gateway_timeout_s,
        connect_timeout=settings.qwen_gateway_connect_timeout_s,
    )
    logger.info(
        "Qwen edit waiting for shared slot",
        extra={
            "event": "qwen_edit_waiting",
            "room_type": result["room_type"],
            "max_concurrent_edits": settings.qwen_max_concurrent_edits,
        },
    )
    async with state.qwen_edit_semaphore:
        logger.info(
            "Qwen edit acquired shared slot",
            extra={
                "event": "qwen_edit_started",
                "room_type": result["room_type"],
                "max_concurrent_edits": settings.qwen_max_concurrent_edits,
            },
        )
        # QwenImageEditClient log đầy đủ prompt ngay trước external request.
        edit_res = await client.mount_products_to_room(
            room_image=image,
            products=candidates,
            room_type=result["room_type"],
            custom_prompt=edit_prompt,
        )

    result["generation_prompt"] = edit_res.prompt
    if edit_res.success and edit_res.generated_image:
        result["generated_image"] = edit_res.generated_image
        result["generation_status"] = "success"
        result["generation_error"] = None
    else:
        result["generation_status"] = "failed"
        result["generation_error"] = edit_res.error
    return RecommendationResponse(**result)


async def _recommend_loaded_image(
    *,
    state: Any,
    settings: Settings,
    image: Any,
    requested_room: str | None,
    top_k: int | None,
    products_per_group: int | None,
    min_price: float | None,
    max_price: float | None,
    confidence: float | None,
    min_score: float | None = None,
    exclude_detected: bool | None = None,
    category_id: str | None = None,
    generate_image: bool | None = None,
    edit_prompt: str | None = None,
    request_slot_reserved: bool = False,
) -> RecommendationResponse:
    """Một pipeline: detect → chọn phòng → recommend → (tuỳ chọn) Qwen."""
    detection_room = requested_room or None
    provided_room = _validate_room(requested_room, state) if requested_room else None

    if request_slot_reserved:
        detected = await asyncio.get_running_loop().run_in_executor(
            state.executor,
            state.detector.detect,
            image,
            confidence if confidence is not None else settings.model_confidence,
            detection_room,
        )
    else:
        if not state.try_acquire():
            raise BusyError("Hệ thống đang xử lý quá nhiều yêu cầu, thử lại sau.")
        try:
            detected = await asyncio.get_running_loop().run_in_executor(
                state.executor,
                state.detector.detect,
                image,
                confidence if confidence is not None else settings.model_confidence,
                detection_room,
            )
        finally:
            state.release()

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
        price_range=_price_range(min_price, max_price, settings.default_min_price),
        category_id=category_id,
    )
    result["room_type_source"] = room_type_source
    result["room_detection_scores"] = room_detection_scores

    should_generate = (
        generate_image if generate_image is not None else settings.auto_generate_image
    )
    if should_generate:
        response = await _generate_recommendation_image(
            state=state,
            settings=settings,
            image=image,
            recommendation=RecommendationResponse(**result),
            edit_prompt=edit_prompt,
        )
    else:
        result["generation_status"] = "skipped"
        response = RecommendationResponse(**result)

    logger.info(
        "recommend done",
        extra={"event": "recommend", "detected": len(detected),
               "groups": len(result["recommendations"]), "room_type": room},
    )
    return response


async def _load_three_images(images: list[UploadFile], settings: Settings) -> list[Any]:
    if len(images) != 3:
        raise ImageError("Cần gửi đúng 3 ảnh với field multipart 'images'.")
    return [await _load_upload_image(image, settings) for image in images]


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
    min_price: float | None = Query(
        None, ge=0, description="Giá thấp nhất (mặc định: 250.000đ).", examples=[250000.0]
    ),
    max_price: float | None = Query(None, ge=0, description="Giá cao nhất."),
    category_id: str | None = Query(
        None,
        description="Chỉ gợi ý trong một nhóm category nội bộ.",
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
    category = _validate_category(category_id)
    _validate_ranking_parameters(top_k, products_per_group)
    loaded_image = await _load_upload_image(image, settings)
    return await _recommend_loaded_image(
        state=state,
        settings=settings,
        image=loaded_image,
        requested_room=requested_room,
        top_k=top_k,
        products_per_group=products_per_group,
        min_price=min_price,
        max_price=max_price,
        confidence=confidence,
        min_score=min_score,
        exclude_detected=exclude_detected,
        category_id=category,
        generate_image=generate_image,
        edit_prompt=edit_prompt,
    )


@router.post(
    "/recommendations/images/auto",
    response_model=ThreeRoomRecommendationResponse,
    summary="Tự nhận diện và gợi ý cho ba ảnh phòng",
    description="Nhận đúng ba ảnh, tự xác định phòng từng ảnh, gợi ý và ghép sản phẩm cho từng phòng.",
    response_description="Ba kết quả theo thứ tự ba ảnh đầu vào",
    operation_id="recommendThreeRoomsAuto",
    responses={
        200: {"description": "Gợi ý và ghép ảnh thành công."},
        400: {"description": "Đầu vào ảnh không hợp lệ.", "content": ERROR_CONTENT},
        404: {"description": "Tham số ngoài miền hợp lệ.", "content": ERROR_CONTENT},
        429: {"description": "Hệ thống đang bận.", "content": ERROR_CONTENT},
        503: {"description": "Model chưa sẵn sàng.", "content": ERROR_CONTENT},
    },
)
async def recommend_three_rooms_auto(
    state: AppStateDep,
    settings: Settings = Depends(get_settings),
    images: list[UploadFile] = File(
        ...,
        description="Gửi đúng ba ảnh bằng ba field multipart cùng tên `images`.",
    ),
    top_k: int | None = Query(None, ge=1, le=TOP_K_MAX, description="Số nhóm mỗi phòng."),
    product_per_group: int | None = Query(
        None,
        ge=1,
        le=PRODUCTS_PER_GROUP_MAX,
        description="Số sản phẩm trong mỗi nhóm của từng phòng.",
    ),
    min_price: float | None = Query(
        None, ge=0, description="Giá thấp nhất (mặc định: 250.000đ).", examples=[250000.0]
    ),
    max_price: float | None = Query(None, ge=0, description="Giá cao nhất."),
    confidence: float | None = Query(None, ge=0, le=1, description="Ngưỡng detector."),
) -> ThreeRoomRecommendationResponse:
    """Ba pipeline song song; DINO tuần tự, Qwen của ảnh trước được overlap."""
    if not state.ready:
        raise ModelNotReadyError("Model chưa sẵn sàng.")
    _validate_ranking_parameters(top_k, product_per_group)
    loaded_images = await _load_three_images(images, settings)
    if not state.try_acquire():
        raise BusyError("Hệ thống đang xử lý quá nhiều yêu cầu, thử lại sau.")
    try:
        results = await asyncio.gather(*[
            _recommend_loaded_image(
                state=state,
                settings=settings,
                image=image,
                requested_room=None,
                top_k=top_k,
                products_per_group=product_per_group,
                min_price=min_price,
                max_price=max_price,
                confidence=confidence,
                generate_image=True,
                request_slot_reserved=True,
            )
            for image in loaded_images
        ])
    finally:
        state.release()
    return ThreeRoomRecommendationResponse(results=results)


@router.post(
    "/recommendations/images/by-room",
    response_model=ThreeRoomRecommendationResponse,
    summary="Gợi ý cho ba ảnh với loại phòng chỉ định",
    description="Nhận đúng ba ảnh và ba room_type tương ứng, gợi ý rồi ghép sản phẩm cho từng phòng.",
    response_description="Ba kết quả theo thứ tự ba ảnh và room_type đầu vào",
    operation_id="recommendThreeRoomsByType",
    responses={
        200: {"description": "Gợi ý và ghép ảnh thành công."},
        400: {"description": "Đầu vào ảnh không hợp lệ.", "content": ERROR_CONTENT},
        404: {"description": "room_type hoặc tham số không hợp lệ.", "content": ERROR_CONTENT},
        429: {"description": "Hệ thống đang bận.", "content": ERROR_CONTENT},
        503: {"description": "Model chưa sẵn sàng.", "content": ERROR_CONTENT},
    },
)
async def recommend_three_rooms_by_type(
    state: AppStateDep,
    settings: Settings = Depends(get_settings),
    images: list[UploadFile] = File(
        ...,
        description="Gửi đúng ba ảnh bằng ba field multipart cùng tên `images`.",
    ),
    room_types: list[str] = Form(
        ...,
        description=(
            "Gửi đúng ba field multipart `room_types`, lần lượt ứng với ba ảnh: "
            "living_room, bedroom hoặc kitchen."
        ),
    ),
    top_k: int | None = Query(None, ge=1, le=TOP_K_MAX, description="Số nhóm mỗi phòng."),
    product_per_group: int | None = Query(
        None,
        ge=1,
        le=PRODUCTS_PER_GROUP_MAX,
        description="Số sản phẩm trong mỗi nhóm của từng phòng.",
    ),
    min_price: float | None = Query(
        None, ge=0, description="Giá thấp nhất (mặc định: 250.000đ).", examples=[250000.0]
    ),
    max_price: float | None = Query(None, ge=0, description="Giá cao nhất."),
    confidence: float | None = Query(None, ge=0, le=1, description="Ngưỡng detector."),
) -> ThreeRoomRecommendationResponse:
    """Ba pipeline song song với room caller chỉ định."""
    if not state.ready:
        raise ModelNotReadyError("Model chưa sẵn sàng.")
    if len(room_types) != 3:
        raise ImageError("Cần gửi đúng 3 room_types tương ứng với 3 ảnh.")
    _validate_ranking_parameters(top_k, product_per_group)
    # Validate hết trước khi chạy inference hoặc gọi gateway.
    validated_rooms = [_validate_room(room_type, state) for room_type in room_types]
    loaded_images = await _load_three_images(images, settings)
    if not state.try_acquire():
        raise BusyError("Hệ thống đang xử lý quá nhiều yêu cầu, thử lại sau.")
    try:
        results = await asyncio.gather(*[
            _recommend_loaded_image(
                state=state,
                settings=settings,
                image=image,
                requested_room=room_type,
                top_k=top_k,
                products_per_group=product_per_group,
                min_price=min_price,
                max_price=max_price,
                confidence=confidence,
                generate_image=True,
                request_slot_reserved=True,
            )
            for image, room_type in zip(loaded_images, validated_rooms, strict=True)
        ])
    finally:
        state.release()
    return ThreeRoomRecommendationResponse(results=results)
