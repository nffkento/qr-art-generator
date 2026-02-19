"""QR Art Generator — AI-powered artistic QR code generation."""

__version__ = "1.0.0"

# Shared constants
TARGET_SIZE = 768  # Optimal for Stable Diffusion 1.5 ControlNet
MAX_QR_DATA_LENGTH = 2953  # Max alphanumeric chars at QR version 40, EC level H
