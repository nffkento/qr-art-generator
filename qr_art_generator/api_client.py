"""Cloud API clients for AI-powered QR code art generation."""

import os
import sys
import time
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Spinner for visual feedback during long API calls
# ---------------------------------------------------------------------------

class Spinner:
    """Simple terminal spinner for long-running operations."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "Generating..."):
        self._message = message
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "Spinner":
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def stop(self, final_message: str = "") -> None:
        self._running = False
        if self._thread:
            self._thread.join()
        # Clear spinner line
        sys.stderr.write(f"\r\033[K")
        if final_message:
            sys.stderr.write(f"  {final_message}\n")
        sys.stderr.flush()

    def _spin(self) -> None:
        idx = 0
        while self._running:
            frame = self.FRAMES[idx % len(self.FRAMES)]
            elapsed = 0
            start = time.time()
            while self._running and time.time() - start < 0.1:
                time.sleep(0.05)
            # Calculate total elapsed since spinner started
            sys.stderr.write(f"\r  {frame} {self._message}")
            sys.stderr.flush()
            idx += 1


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class GenerationParams:
    """Parameters for QR art generation."""

    qr_data: str
    prompt: str
    negative_prompt: str = "ugly, disfigured, low quality, blurry, nsfw"
    guidance_scale: float = 7.5
    controlnet_scale: float = 1.1
    strength: float = 0.9
    seed: int = -1
    num_steps: int = 30
    sampler: str = "DPM++ Karras SDE"
    init_image_path: str | None = None
    qr_image_path: str | None = None
    # IllusionDiffusion-specific
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes


def _retry_with_backoff(
    fn,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    spinner_message: str = "Generating...",
):
    """Execute a function with retry logic, exponential backoff, and a spinner.

    Args:
        fn: Callable to execute.
        max_retries: Maximum number of retry attempts.
        timeout: Per-attempt timeout in seconds.
        spinner_message: Message shown in the spinner.

    Returns:
        The return value of fn().

    Raises:
        The last exception encountered after all retries are exhausted,
        or TimeoutError if the call exceeds the timeout.
    """
    last_exception = None

    for attempt in range(1, max_retries + 1):
        spinner = Spinner(
            f"{spinner_message} (attempt {attempt}/{max_retries})"
            if attempt > 1 else spinner_message
        )
        spinner.start()

        result_container = [None]
        error_container = [None]

        def _run():
            try:
                result_container[0] = fn()
            except Exception as e:
                error_container[0] = e

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            spinner.stop()
            last_exception = TimeoutError(
                f"API call timed out after {timeout}s. "
                "The space may be cold-starting or overloaded."
            )
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  ⏳ Timeout. Retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
            continue

        if error_container[0] is not None:
            spinner.stop()
            last_exception = error_container[0]
            if attempt < max_retries:
                wait = 2 ** attempt
                print(
                    f"  ⚠️  Attempt {attempt} failed: {last_exception}. "
                    f"Retrying in {wait}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)
            continue

        # Success
        spinner.stop()
        return result_container[0]

    raise last_exception  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseAPIClient(ABC):
    """Abstract base class for QR art generation API clients."""

    @abstractmethod
    def generate(self, params: GenerationParams) -> str:
        """Generate an artistic QR code image.

        Args:
            params: Generation parameters.

        Returns:
            Path to the generated image file.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def supports_init_image(self) -> bool:
        """Whether this backend supports img2img with a separate init image."""
        return False


# ---------------------------------------------------------------------------
# HuggingFace client
# ---------------------------------------------------------------------------

class HuggingFaceClient(BaseAPIClient):
    """Client for the HuggingFace QR-code-AI-art-generator space.

    Uses the Gradio client to call the public space. No API key required.
    Space: huggingface-projects/QR-code-AI-art-generator

    Pipeline: StableDiffusionControlNetImg2ImgPipeline
    ControlNet: DionTimmer/controlnet_qrcode-control_v1p_sd15
    """

    SPACE_ID = "huggingface-projects/QR-code-AI-art-generator"

    def __init__(self):
        try:
            from gradio_client import Client
            self._client = Client(self.SPACE_ID)
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to HuggingFace space '{self.SPACE_ID}': {e}\n"
                "Make sure you have internet access and gradio_client installed."
            )

    def name(self) -> str:
        return "HuggingFace (QR-code-AI-art-generator)"

    @property
    def supports_init_image(self) -> bool:
        return True

    def generate(self, params: GenerationParams) -> str:
        from gradio_client import handle_file

        init_image = None
        if params.init_image_path:
            init_image = handle_file(params.init_image_path)

        qrcode_image = None
        if params.qr_image_path:
            qrcode_image = handle_file(params.qr_image_path)

        # When using a custom init image, don't use QR as init image
        use_qr_as_init = params.init_image_path is None

        def _call():
            return self._client.predict(
                # Always send empty string — we provide our own optimized QR image
                qr_code_content="",
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                guidance_scale=params.guidance_scale,
                controlnet_conditioning_scale=params.controlnet_scale,
                strength=params.strength,
                seed=params.seed,
                init_image=init_image,
                qrcode_image=qrcode_image,
                use_qr_code_as_init_image=use_qr_as_init,
                sampler=params.sampler,
                api_name="/inference",
            )

        return _retry_with_backoff(_call, spinner_message="Generating via HuggingFace...")


# ---------------------------------------------------------------------------
# IllusionDiffusion client
# ---------------------------------------------------------------------------

class IllusionDiffusionClient(BaseAPIClient):
    """Client for the AP123/IllusionDiffusion space.

    Higher quality output (1024x1024) with a two-stage pipeline.
    Requires a pre-generated QR code image as input.
    Does NOT support a separate init image for img2img.

    Pipeline: StableDiffusionControlNetPipeline (512) + Img2Img upscale (1024)
    ControlNet: monster-labs/control_v1p_sd15_qrcode_monster
    Base: SG161222/Realistic_Vision_V5.1_noVAE
    """

    SPACE_ID = "AP123/IllusionDiffusion"

    def __init__(self):
        try:
            from gradio_client import Client
            self._client = Client(self.SPACE_ID)
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to HuggingFace space '{self.SPACE_ID}': {e}\n"
                "Make sure you have internet access and gradio_client installed."
            )

    def name(self) -> str:
        return "IllusionDiffusion (1024x1024, higher quality)"

    @property
    def supports_init_image(self) -> bool:
        return False

    def generate(self, params: GenerationParams) -> str:
        from gradio_client import handle_file

        if not params.qr_image_path:
            raise ValueError(
                "IllusionDiffusion requires a pre-generated QR code image. "
                "Provide qr_image_path in params."
            )

        def _call():
            return self._client.predict(
                control_image=handle_file(params.qr_image_path),
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                guidance_scale=params.guidance_scale,
                controlnet_conditioning_scale=params.controlnet_scale,
                control_guidance_start=params.control_guidance_start,
                control_guidance_end=params.control_guidance_end,
                upscaler_strength=params.strength,
                seed=params.seed,
                sampler=params.sampler if params.sampler in ("Euler", "DPM++ Karras SDE") else "Euler",
                api_name="/inference",
            )

        result = _retry_with_backoff(_call, spinner_message="Generating via IllusionDiffusion...")

        # IllusionDiffusion returns (image_path, visibility_dict, seed)
        if isinstance(result, (list, tuple)):
            return result[0]
        return result


# ---------------------------------------------------------------------------
# Replicate client
# ---------------------------------------------------------------------------

class ReplicateClient(BaseAPIClient):
    """Client for Replicate API. Requires REPLICATE_API_TOKEN env var."""

    def __init__(self):
        self._token = os.environ.get("REPLICATE_API_TOKEN")
        if not self._token:
            raise ValueError(
                "REPLICATE_API_TOKEN environment variable not set.\n"
                "Get your token at https://replicate.com/account/api-tokens"
            )
        try:
            import replicate
            self._replicate = replicate
        except ImportError:
            raise ImportError("replicate package not installed. Run: pip install replicate")

    def name(self) -> str:
        return "Replicate API"

    @property
    def supports_init_image(self) -> bool:
        return False

    def generate(self, params: GenerationParams) -> str:
        def _call():
            return self._replicate.run(
                "qr2ai/qr_code_ai_art_generator",
                input={
                    "url": params.qr_data,
                    "prompt": params.prompt,
                    "negative_prompt": params.negative_prompt,
                    "guidance_scale": params.guidance_scale,
                    "controlnet_conditioning_scale": params.controlnet_scale,
                    "strength": params.strength,
                    "seed": params.seed if params.seed >= 0 else None,
                    "num_inference_steps": params.num_steps,
                },
            )

        output = _retry_with_backoff(_call, spinner_message="Generating via Replicate...")

        # Replicate returns a URL or list of URLs
        if isinstance(output, list):
            url = output[0]
        else:
            url = output

        # Download the image with a timeout
        import urllib.request
        tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="qr_art_replicate_", delete=False)
        urllib.request.urlretrieve(str(url), tmp.name, reporthook=None)
        tmp.close()
        return tmp.name


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_client(api: str = "huggingface") -> BaseAPIClient:
    """Factory function to get the appropriate API client.

    Args:
        api: One of "huggingface", "illusion", or "replicate".

    Returns:
        An initialized API client.
    """
    clients = {
        "huggingface": HuggingFaceClient,
        "illusion": IllusionDiffusionClient,
        "replicate": ReplicateClient,
    }

    if api not in clients:
        raise ValueError(f"Unknown API '{api}'. Choose from: {', '.join(clients.keys())}")

    return clients[api]()
