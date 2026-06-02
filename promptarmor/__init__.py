"""PromptArmor — a self-hosted, LLM-prompted prompt-injection content guard.

Independent reimplementation of the method in arXiv:2507.15219 (Shi et al.).
Not affiliated with the paper authors, and unrelated to the GitHub project
`prompt-armor/prompt-armor` (a regex/meta-classifier with the same name).
"""

from .config import PAConfig
from .client import GuardrailClient
from .detector import Detector, parse_output
from .sanitize import sanitize, fuzzy_remove_one
from .api import PromptArmor

__version__ = "0.1.0"

__all__ = [
    "PAConfig",
    "GuardrailClient",
    "Detector",
    "parse_output",
    "sanitize",
    "fuzzy_remove_one",
    "PromptArmor",
    "__version__",
]
