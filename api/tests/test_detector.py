"""Unit tests cho batch Grounding DINO, không cần nạp checkpoint thật."""

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
    labels = ("sofa", "bed", "refrigerator")

    def __init__(self):
        self.calls: list[tuple[list[Image.Image], list[str]]] = []
        self.target_sizes: list[list[tuple[int, int]]] = []

    def __call__(self, *, images, text, return_tensors: str, **kwargs):
        assert return_tensors == "pt"
        assert kwargs["images_kwargs"]["size"] == {
            "shortest_edge": 512,
            "longest_edge": 768,
        }
        self.calls.append((list(images), list(text)))
        return _FakeInputs(pixel_values="fake")

    def post_process_grounded_object_detection(self, **kwargs):
        self.target_sizes.append(kwargs["target_sizes"])
        return [
            {"text_labels": [label], "scores": [0.9 - (index * 0.1)]}
            for index, label in enumerate(self.labels)
        ]


def test_detect_batch_keeps_each_images_labels_separate() -> None:
    """Một batch dùng prompt chung nhưng kết quả phải giữ đúng image index."""
    detector = Detector(device="cpu", jpeg_roundtrip=False, batch_size=3)
    processor = _FakeProcessor()
    detector._processor = processor
    detector._model = _FakeModel()

    images = [
        Image.new("RGB", (20, 10)),
        Image.new("RGB", (40, 30)),
        Image.new("RGB", (60, 50)),
    ]
    detected = detector.detect_batch(images, confidence=0.5)

    assert [[item["label"] for item in result] for result in detected] == [
        ["Sofa"],
        ["Bed"],
        ["Refrigerator"],
    ]
    assert len(processor.calls) == len(detector.room_prompts["default"])
    assert all(len(batch_images) == 3 for batch_images, _ in processor.calls)
    assert all(len(set(batch_text)) == 1 for _, batch_text in processor.calls)
    assert processor.target_sizes[0] == [(10, 20), (30, 40), (50, 60)]


def test_auto_batch_size_avoids_large_batch_on_cpu() -> None:
    """Chỉ CUDA mặc định dùng batch 3; CPU tránh padding batch chậm."""
    assert Detector(device="cpu").inference_batch_size == 1
    assert Detector(device="cuda").inference_batch_size == 3
