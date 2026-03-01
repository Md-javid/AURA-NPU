"""
Aura-NPU app/utils package.
Exposes core utility modules for the main application.
"""

from app.utils.screen_capture import ScreenCapture
from app.utils.language_config import LANGUAGE_MAP, get_language_names, get_prompt_suffix

__all__ = ["ScreenCapture", "LANGUAGE_MAP", "get_language_names", "get_prompt_suffix"]
