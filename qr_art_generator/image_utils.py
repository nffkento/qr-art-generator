"""Image processing utilities for QR art generation."""

import os
import shutil
import tempfile
from enum import Enum
from PIL import Image

from qr_art_generator import TARGET_SIZE


class VerifyResult(Enum):
    """Result of QR scannability verification."""
    SCANNABLE = "scannable"
    NOT_SCANNABLE = "not_scannable"
    SKIPPED = "skipped"  # pyzbar not installed


def load_and_resize_image(
    path: str,
    size: int = TARGET_SIZE,
    preserve_aspect: bool = True,
) -> Image.Image:
    """Load an image from disk and resize to target dimensions.

    Args:
        path: Path to the input image file.
        size: Target size (square) in pixels.
        preserve_aspect: If True, center-crop to square before resizing
            instead of stretching. Default True.

    Returns:
        Resized PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the image file doesn't exist.
        ValueError: If the file is not a valid image.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        img = Image.open(path)
        img = img.convert("RGB")

        if preserve_aspect:
            img = _center_crop_square(img)

        img = img.resize((size, size), Image.LANCZOS)
        return img
    except (OSError, SyntaxError) as e:
        raise ValueError(f"Could not open image '{path}': {e}")


def _center_crop_square(img: Image.Image) -> Image.Image:
    """Center-crop an image to a square, preserving aspect ratio.

    Takes the largest centered square region from the image.
    """
    width, height = img.size
    if width == height:
        return img

    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def save_init_image_temp(img: Image.Image) -> str:
    """Save an init image to a unique temporary file.

    Uses tempfile to avoid race conditions and ensure cross-platform compatibility.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="qr_art_init_", delete=False)
    img.save(tmp.name, "PNG")
    tmp.close()
    return tmp.name


def save_output(source_path: str, output_path: str) -> str:
    """Copy the generated image from the API temp path to the user's desired output.

    Handles format conversion if the source and output extensions differ
    (e.g., .webp -> .png).

    Args:
        source_path: Path to the generated image (from API response).
        output_path: Desired output path.

    Returns:
        The output path where the image was saved.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    src_ext = os.path.splitext(source_path)[1].lower()
    dst_ext = os.path.splitext(output_path)[1].lower()

    if src_ext != dst_ext or source_path == output_path:
        # Convert format via PIL
        img = Image.open(source_path)
        img.save(output_path)
    else:
        shutil.copy2(source_path, output_path)

    return output_path


def verify_qr_scannable(image_path: str) -> tuple[VerifyResult, str | None]:
    """Attempt to decode a QR code from the generated image.

    Uses pyzbar if available, otherwise returns SKIPPED.

    Args:
        image_path: Path to the image to verify.

    Returns:
        Tuple of (VerifyResult, decoded_data: str | None).
    """
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode

        img = Image.open(image_path)
        results = pyzbar_decode(img)
        if results:
            decoded = results[0].data.decode("utf-8")
            return VerifyResult.SCANNABLE, decoded
        return VerifyResult.NOT_SCANNABLE, None
    except ImportError:
        return VerifyResult.SKIPPED, None
    except Exception:
        return VerifyResult.NOT_SCANNABLE, None


def cleanup_temp_files(*paths: str) -> None:
    """Remove temporary files, silently ignoring errors."""
    for path in paths:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
