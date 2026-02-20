"""CLI entry point for QR Art Generator."""

import argparse
import os
import sys
import time

from qr_art_generator import __version__


# Sentinel to detect if user explicitly set --controlnet-scale
_CONTROLNET_SCALE_DEFAULT = object()

# Default negative prompt
_DEFAULT_NEGATIVE = "ugly, disfigured, low quality, blurry, nsfw"


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: str = "") -> str:
    """Prompt the user for input, showing a default value."""
    if default:
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw if raw else default
    else:
        while True:
            raw = input(f"{prompt}: ").strip()
            if raw:
                return raw
            print("  ⚠  入力が必要です。もう一度入力してください。")


def _ask_choice(prompt: str, choices: list[str], default: int = 1) -> int:
    """Prompt the user to select from numbered choices. Returns 1-based index."""
    print(prompt)
    for i, choice in enumerate(choices, 1):
        marker = " ← デフォルト" if i == default else ""
        print(f"  [{i}] {choice}{marker}")
    while True:
        raw = input(f"> ").strip()
        if not raw:
            return default
        try:
            n = int(raw)
            if 1 <= n <= len(choices):
                return n
        except ValueError:
            pass
        print(f"  ⚠  1〜{len(choices)} の数字を入力してください。")


def interactive_mode() -> list[str]:
    """Walk the user through QR art generation interactively.

    Returns:
        An argv list suitable for argparse (e.g. ['--url', '...', '--prompt', '...']).
    """
    from qr_art_generator.local_blend import STYLE_DESCRIPTIONS, STYLE_ORDER

    print(f"\n🎨 QR Art Generator v{__version__} — Interactive Mode")
    print("=" * 50)

    argv: list[str] = []

    # --- URL ---
    print()
    url = _ask("📎 QRコードに埋め込むURLを入力してください")
    argv.extend(["--url", url])

    # --- Mode ---
    print()
    mode = _ask_choice(
        "🖼️  モードを選択してください:",
        [
            "テキストからQRアート生成 (AIでプロンプトから画像を作る)",
            "画像にQRコードを埋め込む (既存画像をローカルで合成)",
        ],
        default=1,
    )

    if mode == 2:
        # --- Image blending mode (local, no AI) ---
        print()
        image_path = _ask("  📁 画像ファイルのパスを入力してください")
        argv.extend(["--image", image_path])

        # Style selection
        print()
        style_labels = [STYLE_DESCRIPTIONS[s] for s in STYLE_ORDER]
        style_choice = _ask_choice(
            "🎨 ブレンドスタイルを選択してください:",
            style_labels,
            default=1,
        )
        selected_style = STYLE_ORDER[style_choice - 1]
        argv.extend(["--blend-style", selected_style.value])

        # Optional logo
        print()
        logo = input("🏷️  中央にロゴを入れますか？ パスを入力 (空欄でスキップ): ").strip()
        if logo:
            argv.extend(["--logo", logo])

        # No prompt needed for local blending — use a dummy
        argv.extend(["--prompt", "(local blend)"])

    else:
        # --- AI text-to-QR-art mode ---
        # --- Prompt ---
        print()
        prompt = _ask("✏️  画像のプロンプトを入力してください (例: \"Japanese zen garden, cherry blossoms\")")
        argv.extend(["--prompt", prompt])

        # --- API backend ---
        print()
        api_choice = _ask_choice(
            "🔧 APIバックエンドを選択してください:",
            [
                "HuggingFace (768x768, 高速)",
                "IllusionDiffusion (1024x1024, 高画質)",
                "Replicate (APIキー必要)",
            ],
            default=1,
        )
        api_map = {1: "huggingface", 2: "illusion", 3: "replicate"}
        argv.extend(["--api", api_map[api_choice]])

    # --- Output ---
    print()
    output = _ask("💾 出力ファイル名", default="qr_art_output.png")
    # Auto-append .png if user didn't include an extension
    if "." not in os.path.basename(output):
        output += ".png"
    argv.extend(["-o", output])

    # --- Advanced settings (AI mode only) ---
    if mode == 1:
        print()
        advanced = input("⚙️  詳細設定を変更しますか? [y/N]: ").strip().lower()
        if advanced in ("y", "yes"):
            # ControlNet scale
            default_cn = "2.0" if api_choice == 2 else "1.1"
            cn = _ask(f"  ControlNet scale (0.5-2.0)", default=default_cn)
            argv.extend(["--controlnet-scale", cn])

            # Strength
            strength = _ask("  Strength (0.0-1.0)", default="0.9")
            argv.extend(["--strength", strength])

            # Seed
            seed = _ask("  Seed (-1=ランダム)", default="-1")
            argv.extend(["--seed", seed])

            # Negative prompt
            neg = _ask("  Negative prompt", default=_DEFAULT_NEGATIVE)
            argv.extend(["--negative-prompt", neg])

    # Always overwrite from interactive mode (user just decided the filename)
    argv.append("--overwrite")

    print()
    print("─" * 50)
    if mode == 2:
        print("→ ローカルでQRコードを合成します...")
    else:
        print("→ AI生成を開始します...")
    print()

    return argv


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qr-art-generator",
        description="AI-powered artistic QR code generator. Turns URLs into scannable art.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for beginners)
  qr-art

  # Local blend: image as background behind QR
  qr-art --url "https://example.com" --image photo.jpg --prompt _ --blend-style background

  # Local blend: dot-style QR colored by image
  qr-art --url "https://example.com" --image photo.jpg --prompt _ --blend-style dots

  # AI mode: text-to-QR-Art (generate art from a prompt)
  qr-art --url "https://example.com" --prompt "Japanese zen garden, watercolor"

  # AI mode: IllusionDiffusion for higher quality (1024x1024)
  qr-art --url "https://example.com" --prompt "medieval castle" --api illusion

  # AI mode: fine-tune parameters
  qr-art --url "https://example.com" --prompt "cyberpunk city" \\
    --controlnet-scale 1.3 --seed 42 --steps 40
        """,
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Required
    parser.add_argument(
        "--url",
        required=True,
        help="URL or text to encode in the QR code",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help='Text prompt for AI image generation (e.g., "Japanese garden, watercolor"). '
             'Ignored in local blend mode (--image).',
    )

    # Optional — input/output
    parser.add_argument(
        "--image",
        default=None,
        help="Path to an input image. Enables LOCAL blend mode (no AI, instant).",
    )
    parser.add_argument(
        "--blend-style",
        default="background",
        choices=["background", "dots", "rounded", "squares", "bars_vertical", "bars_horizontal"],
        help="Blend style for local image mode (default: background). "
             "Only used with --image.",
    )
    parser.add_argument(
        "--logo",
        default=None,
        help="Path to a logo image to embed in the center of the QR code. "
             "Only used with --image + styled modes (not background).",
    )
    parser.add_argument(
        "--output", "-o",
        default="qr_art_output.png",
        help="Output image path (default: qr_art_output.png)",
    )

    # Optional — generation parameters
    parser.add_argument(
        "--controlnet-scale",
        type=float,
        default=_CONTROLNET_SCALE_DEFAULT,
        help="ControlNet conditioning scale (0.5-2.0). Higher = more scannable, less artistic. "
             "Default: 1.1 for huggingface, 2.0 for illusion",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (1.0-20.0). Higher = stronger prompt adherence. Default: 7.5",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.9,
        help="Denoising strength (0.0-1.0). Higher = more creative. Default: 0.9",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for reproducibility. -1 for random. Default: -1",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps (Replicate only, others use space defaults). Default: 30",
    )
    parser.add_argument(
        "--negative-prompt",
        default="ugly, disfigured, low quality, blurry, nsfw",
        help="Negative prompt (things to avoid in generation)",
    )
    parser.add_argument(
        "--sampler",
        default="DPM++ Karras SDE",
        choices=["DPM++ Karras SDE", "DPM++ Karras", "Heun", "Euler", "DDIM", "DEIS"],
        help="Diffusion sampler. Default: DPM++ Karras SDE",
    )

    # IllusionDiffusion-specific
    parser.add_argument(
        "--control-start",
        type=float,
        default=0.0,
        help="ControlNet guidance start (0.0-1.0). When ControlNet kicks in. "
             "Only used with --api illusion. Default: 0.0",
    )
    parser.add_argument(
        "--control-end",
        type=float,
        default=1.0,
        help="ControlNet guidance end (0.0-1.0). When ControlNet stops. "
             "Only used with --api illusion. Default: 1.0",
    )

    # API selection
    parser.add_argument(
        "--api",
        default="huggingface",
        choices=["huggingface", "illusion", "replicate"],
        help="Which cloud API to use. Default: huggingface",
    )

    # Flags
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip QR code scannability verification of the output",
    )
    parser.add_argument(
        "--white-bg",
        action="store_true",
        help="Use white background for QR code instead of gray (may reduce blending quality)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file without prompting",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    # If no arguments given, launch interactive mode
    if argv is None and len(sys.argv) <= 1:
        argv = interactive_mode()

    parser = create_parser()
    args = parser.parse_args(argv)

    # Lazy imports for faster --help
    from qr_art_generator.image_utils import (
        save_output, verify_qr_scannable, cleanup_temp_files, VerifyResult,
    )

    print(f"QR Art Generator v{__version__}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # Check output overwrite
    # ------------------------------------------------------------------
    if os.path.exists(args.output) and not args.overwrite:
        response = input(f"  Output file '{args.output}' already exists. Overwrite? [y/N] ")
        if response.lower() not in ("y", "yes"):
            print("  Aborted.")
            return 0

    # ------------------------------------------------------------------
    # Branch: Local blend mode vs AI generation mode
    # ------------------------------------------------------------------
    if args.image:
        return _run_local_blend(args)
    else:
        return _run_ai_generation(args)


def _run_local_blend(args) -> int:
    """Run local image + QR code blending (no AI, instant)."""
    from qr_art_generator.local_blend import BlendStyle, blend_qr_with_image
    from qr_art_generator.image_utils import verify_qr_scannable, VerifyResult

    print(f"\n🖼️  ローカルブレンドモード (AI不要・即時生成)")
    print(f"  Image:  {args.image}")
    print(f"  Style:  {args.blend_style}")
    if args.logo:
        print(f"  Logo:   {args.logo}")
    print()

    try:
        style = BlendStyle(args.blend_style)
    except ValueError:
        print(f"  ERROR: Unknown blend style: {args.blend_style}", file=sys.stderr)
        return 1

    start_time = time.time()
    try:
        output_path = blend_qr_with_image(
            data=args.url,
            image_path=args.image,
            style=style,
            output_path=args.output,
            logo_path=args.logo,
        )
        elapsed = time.time() - start_time
        print(f"  ✓ ブレンド完了 ({elapsed:.2f}s)")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ERROR: {e}", file=sys.stderr)
        return 1

    print(f"  ✓ Saved: {output_path}")

    if not args.no_verify:
        print(f"\n  Verifying QR code scannability...")
        result, decoded = verify_qr_scannable(output_path)
        if result == VerifyResult.SCANNABLE:
            print(f"  ✓ QR code is SCANNABLE! Decoded: {decoded}")
        elif result == VerifyResult.SKIPPED:
            print(f"  ⊘ Verification skipped (pyzbar not installed)")
            print(f"    Install with: pip install pyzbar")
        else:
            print(f"  ⚠️  WARNING: QR code may not be scannable.")
            print(f"     Try a different --blend-style or simpler image.")

    print(f"\n✅ Done! Your QR code is at: {output_path}")
    return 0


def _run_ai_generation(args) -> int:
    """Run AI-powered QR art generation via cloud API."""
    from qr_art_generator.qr_generator import generate_qr_code, save_qr_temp
    from qr_art_generator.api_client import get_client, GenerationParams
    from qr_art_generator.image_utils import (
        save_output, verify_qr_scannable, cleanup_temp_files, VerifyResult,
    )

    # ------------------------------------------------------------------
    # Resolve smart defaults for controlnet_scale
    # ------------------------------------------------------------------
    user_set_controlnet = args.controlnet_scale is not _CONTROLNET_SCALE_DEFAULT
    if not user_set_controlnet:
        if args.api == "illusion":
            args.controlnet_scale = 2.0
        else:
            args.controlnet_scale = 1.1

    # Track temp files for cleanup
    temp_files: list[str] = []

    try:
        # Step 1: Generate base QR code
        print(f"\n[1/3] Generating base QR code for: {args.url}")
        qr_image = generate_qr_code(
            data=args.url,
            use_gray_background=not args.white_bg,
        )
        qr_path = save_qr_temp(qr_image)
        temp_files.append(qr_path)
        print(f"  ✓ Base QR code ready")

        # Step 2: Call cloud API
        print(f"\n[2/3] Generating artistic QR code via {args.api} API...")
        print(f"  Prompt:          {args.prompt}")
        print(f"  ControlNet:      {args.controlnet_scale}")
        print(f"  Guidance:        {args.guidance_scale}")
        print(f"  Strength:        {args.strength}")
        print(f"  Seed:            {'random' if args.seed == -1 else args.seed}")
        print(f"  Sampler:         {args.sampler}")
        if args.api == "illusion":
            print(f"  Control start:   {args.control_start}")
            print(f"  Control end:     {args.control_end}")
        print()

        try:
            client = get_client(args.api)
            print(f"  Backend: {client.name()}")
        except (ConnectionError, ValueError, ImportError) as e:
            print(f"\n  ERROR: {e}", file=sys.stderr)
            return 1

        params = GenerationParams(
            qr_data=args.url,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            controlnet_scale=args.controlnet_scale,
            strength=args.strength,
            seed=args.seed,
            num_steps=args.steps,
            sampler=args.sampler,
            init_image_path=None,
            qr_image_path=qr_path,
            control_guidance_start=args.control_start,
            control_guidance_end=args.control_end,
        )

        start_time = time.time()
        try:
            result_path = client.generate(params)
            elapsed = time.time() - start_time
            print(f"  ✓ Generation completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"\n  ERROR during generation: {e}", file=sys.stderr)
            return 1

        # Step 3: Save and verify
        print(f"\n[3/3] Saving output to: {args.output}")
        output_path = save_output(result_path, args.output)
        print(f"  ✓ Saved: {output_path}")

        if not args.no_verify:
            print(f"\n  Verifying QR code scannability...")
            result, decoded = verify_qr_scannable(output_path)
            if result == VerifyResult.SCANNABLE:
                print(f"  ✓ QR code is SCANNABLE! Decoded: {decoded}")
            elif result == VerifyResult.SKIPPED:
                print(f"  ⊘ Verification skipped (pyzbar not installed)")
                print(f"    Install with: pip install pyzbar")
            else:
                print(f"  ⚠️  WARNING: QR code may not be scannable.")
                print(f"     Try increasing --controlnet-scale (current: {args.controlnet_scale})")
                print(f"     Or decreasing --strength (current: {args.strength})")

        print(f"\n✅ Done! Your artistic QR code is at: {output_path}")
        return 0

    except (ValueError, FileNotFoundError) as e:
        print(f"\n  ERROR: {e}", file=sys.stderr)
        return 1

    finally:
        # Clean up temp files
        cleanup_temp_files(*temp_files)


if __name__ == "__main__":
    sys.exit(main())
