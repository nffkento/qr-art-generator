"""Local QR code + image blending without AI.

Uses python-qrcode (StyledPilImage) and segno (qrcode-artistic) to create
styled QR codes with images blended in — instantly, offline, no API needed.
"""

import os
import tempfile
from enum import Enum

import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers.pil import (
    CircleModuleDrawer,
    GappedSquareModuleDrawer,
    RoundedModuleDrawer,
    SquareModuleDrawer,
    VerticalBarsDrawer,
    HorizontalBarsDrawer,
)
from qrcode.image.styles.colormasks import ImageColorMask, SolidFillColorMask
from PIL import Image


class BlendStyle(Enum):
    """Available QR code blending styles."""

    # Image colors the dark modules (python-qrcode ImageColorMask)
    DOTS = "dots"                 # Circle modules colored by image
    ROUNDED = "rounded"           # Rounded square modules colored by image
    SQUARES = "squares"           # Standard squares colored by image
    BARS_V = "bars_vertical"      # Vertical bars colored by image
    BARS_H = "bars_horizontal"    # Horizontal bars colored by image

    # Image visible in background (segno artistic)
    BACKGROUND = "background"     # Full image as background behind QR


# Human-readable descriptions for interactive mode
STYLE_DESCRIPTIONS = {
    BlendStyle.DOTS: "ドット (丸いモジュール + 画像カラー)",
    BlendStyle.ROUNDED: "角丸 (角丸モジュール + 画像カラー)",
    BlendStyle.SQUARES: "スクエア (四角モジュール + 画像カラー)",
    BlendStyle.BARS_V: "縦バー (縦ストライプ + 画像カラー)",
    BlendStyle.BARS_H: "横バー (横ストライプ + 画像カラー)",
    BlendStyle.BACKGROUND: "背景ブレンド (画像の上にQRを重ねる)",
}

# Ordered list for interactive display
STYLE_ORDER = [
    BlendStyle.BACKGROUND,
    BlendStyle.DOTS,
    BlendStyle.ROUNDED,
    BlendStyle.SQUARES,
    BlendStyle.BARS_V,
    BlendStyle.BARS_H,
]

_MODULE_DRAWERS = {
    BlendStyle.DOTS: CircleModuleDrawer,
    BlendStyle.ROUNDED: RoundedModuleDrawer,
    BlendStyle.SQUARES: GappedSquareModuleDrawer,
    BlendStyle.BARS_V: VerticalBarsDrawer,
    BlendStyle.BARS_H: HorizontalBarsDrawer,
}


def blend_qr_with_image(
    data: str,
    image_path: str,
    style: BlendStyle = BlendStyle.BACKGROUND,
    output_path: str = "qr_blend_output.png",
    logo_path: str | None = None,
    invert: bool = False,
    box_size: int = 30,
    scale: int = 32,
) -> str:
    """Create a styled QR code blended with an image.

    Args:
        data: URL or text to encode.
        image_path: Path to the image to blend.
        style: The blending style to use.
        output_path: Where to save the result.
        logo_path: Optional center logo image path.
        invert: If True, swap dark/light (black bg + white/image modules).
        box_size: Module size for python-qrcode styles (default 30 → ~1350px).
        scale: Scale factor for segno background style (default 32 → ~1184px).

    Returns:
        The output path where the image was saved.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if style == BlendStyle.BACKGROUND:
        return _blend_segno_background(data, image_path, output_path, scale, invert)
    else:
        return _blend_qrcode_styled(data, image_path, style, output_path, logo_path, box_size, invert)


def _blend_segno_background(
    data: str,
    image_path: str,
    output_path: str,
    scale: int,
    invert: bool = False,
) -> str:
    """Blend using segno: image as background, QR pattern on top."""
    import segno

    qr = segno.make(data, error="h")
    if invert:
        qr.to_artistic(
            background=image_path,
            target=output_path,
            scale=scale,
            dark="#fff",
            light="#000",
        )
    else:
        qr.to_artistic(
            background=image_path,
            target=output_path,
            scale=scale,
        )
    return output_path


def _blend_qrcode_styled(
    data: str,
    image_path: str,
    style: BlendStyle,
    output_path: str,
    logo_path: str | None,
    box_size: int,
    invert: bool = False,
) -> str:
    """Blend using python-qrcode: image colors the modules."""
    qr = qrcode.QRCode(
        error_correction=qrcode.ERROR_CORRECT_H,
        box_size=box_size,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    drawer_class = _MODULE_DRAWERS.get(style, GappedSquareModuleDrawer)

    if invert:
        # Invert: black background, image colors the LIGHT modules
        # We invert the QR matrix so python-qrcode draws the "light" areas
        # as the "dark" modules — then apply the image color to those.
        for r in range(len(qr.modules)):
            for c in range(len(qr.modules[r])):
                qr.modules[r][c] = not qr.modules[r][c]
        kwargs = {
            "image_factory": StyledPilImage,
            "module_drawer": drawer_class(),
            "color_mask": ImageColorMask(
                back_color=(0, 0, 0),
                color_mask_path=image_path,
            ),
        }
    else:
        kwargs = {
            "image_factory": StyledPilImage,
            "module_drawer": drawer_class(),
            "color_mask": ImageColorMask(
                back_color=(255, 255, 255),
                color_mask_path=image_path,
            ),
        }

    if logo_path and os.path.exists(logo_path):
        kwargs["embeded_image_path"] = logo_path

    img = qr.make_image(**kwargs)
    img.save(output_path)
    return output_path
