"""Image input validation for the cat and dog classifier."""

from io import BytesIO

from PIL import Image, UnidentifiedImageError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_FORMATS = {"JPEG", "PNG"}


class InvalidImageError(ValueError):
    """Raised when an uploaded file is not a supported image."""


def decode_image(data: bytes) -> Image.Image:
    """Validate uploaded bytes and return a normalized RGB image."""
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidImageError("Image must be 10 MB or smaller.")

    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in SUPPORTED_FORMATS:
                raise InvalidImageError("Please upload a JPEG or PNG image.")
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("The uploaded file is not a valid image.") from exc
