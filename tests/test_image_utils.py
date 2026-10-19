"""Tests for image upload validation."""

from io import BytesIO

import pytest
from PIL import Image

from app.image_utils import MAX_IMAGE_BYTES, InvalidImageError, decode_image


def make_image_bytes(image_format: str) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color="orange").save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
def test_valid_images_are_normalized_to_rgb(image_format: str) -> None:
    result = decode_image(make_image_bytes(image_format))

    assert result.mode == "RGB"
    assert result.size == (8, 8)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n"])
def test_empty_or_corrupt_uploads_are_rejected(data: bytes) -> None:
    with pytest.raises(InvalidImageError):
        decode_image(data)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (make_image_bytes("GIF"), "JPEG or PNG"),
        (b"x" * (MAX_IMAGE_BYTES + 1), "10 MB or smaller"),
    ],
)
def test_unsupported_or_oversized_uploads_are_rejected(
    data: bytes, message: str
) -> None:
    with pytest.raises(InvalidImageError, match=message):
        decode_image(data)
