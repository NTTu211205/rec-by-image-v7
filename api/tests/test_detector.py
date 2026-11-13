"""Unit tests cho Grounding DINO, không cần nạp checkpoint thật."""

from __future__ import annotations

from PIL import Image

from app.services.detector import Detector


class _FakeInputs(dict):
    input_ids = object()

    def to(self, _device: str):
        return self


class _FakeModel:
    def __call__(self, **_inputs):
        return object()


class _FakeProcessor:
    def __init__(self):
        self.calls: list[tuple[Image.Image, str]] = []
        self.target_sizes: list[list[tuple[int, int]]] = []

    def __call__(self, *, images, text, return_tensors: str, **kwargs):
        assert return_tensors == "pt"
        assert kwargs["images_kwargs"]["size"] == {
            "shortest_edge": 384,
            "longest_edge": 640,
        }
        self.calls.append((images, text))
        return _FakeInputs(pixel_values="fake")

    def post_process_grounded_object_detection(self, **kwargs):
        self.target_sizes.append(kwargs["target_sizes"])
        return [{"text_labels": ["sofa"], "scores": [0.9]}]


def test_detect_uses_single_image_and_configured_resize() -> None:
    """Detector một ảnh giữ resize 512/768, không tạo Grounding DINO batch."""
    detector = Detector(device="cpu", jpeg_roundtrip=False)
    processor = _FakeProcessor()
    detector._processor = processor
    detector._model = _FakeModel()

    detected = detector.detect(Image.new("RGB", (20, 10)), confidence=0.5)

    assert [item["label"] for item in detected] == ["Sofa"]
    assert len(processor.calls) == len(detector.room_prompts["default"])
    assert all(isinstance(image, Image.Image) for image, _ in processor.calls)
    assert all(isinstance(text, str) for _, text in processor.calls)
    assert processor.target_sizes[0] == [(10, 20)]
