"""Generate base QR codes optimized for artistic AI processing."""

import tempfile
import qrcode
from PIL import Image

from qr_art_generator import TARGET_SIZE, MAX_QR_DATA_LENGTH


def generate_qr_code(
    data: str,
    size: int = TARGET_SIZE,
    use_gray_background: bool = True,
) -> Image.Image:
    """Generate a QR code image optimized for ControlNet-based art generation.

    Uses error correction level H (30% redundancy) to maximize scannability
    even after heavy artistic transformation. Optionally uses gray background
    (#808080) which helps ControlNet blend the QR pattern more naturally.

    Args:
        data: The text or URL to encode in the QR code.
        size: Output image size in pixels (square). Default 768 for SD 1.5.
        use_gray_background: Use gray (#808080) instead of white for better blending.

    Returns:
        PIL Image of the QR code at the specified size.

    Raises:
        ValueError: If the data exceeds QR code capacity.
    """
    if len(data) > MAX_QR_DATA_LENGTH:
        raise ValueError(
            f"QR data too long ({len(data)} chars). "
            f"Maximum is {MAX_QR_DATA_LENGTH} characters with error correction level H."
        )

    if not data.strip():
        raise ValueError("QR data cannot be empty.")

    back_color = "#808080" if use_gray_background else "white"

    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    qr_image = qr.make_image(fill_color="black", back_color=back_color)
    qr_image = qr_image.convert("RGB")
    qr_image = qr_image.resize((size, size), Image.LANCZOS)

    return qr_image


def save_qr_temp(qr_image: Image.Image) -> str:
    """Save QR code to a unique temporary file and return the path.

    Uses tempfile to avoid race conditions and ensure cross-platform compatibility.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="qr_art_base_", delete=False)
    qr_image.save(tmp.name, "PNG")
    tmp.close()
    return tmp.name
