from .config import PAConfig
from .client import GuardrailClient
from .detector import Detector
from .sanitize import sanitize


class PromptArmor:
    def __init__(self, cfg: PAConfig | None = None):
        self.cfg = cfg or PAConfig()
        self.det = Detector(self.cfg, GuardrailClient(self.cfg))

    def __call__(self, data: str, user_task: str | None = None) -> dict:
        d = self.det.detect(data, user_task)
        return sanitize(data, d, self.cfg)
