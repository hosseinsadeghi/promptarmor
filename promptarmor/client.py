import re

from openai import OpenAI

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class GuardrailClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def _extra_body(self):
        body = {}
        if not self.cfg.enable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if self.cfg.use_guided_json:
            from .prompts import JSON_SCHEMA
            body["guided_json"] = JSON_SCHEMA   # vLLM guided decoding
        return body

    def complete(self, system: str, data: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model, temperature=self.cfg.temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": data}],
            extra_body=self._extra_body(),
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        # prefer separated reasoning if the server provides it; else strip <think>
        if not getattr(msg, "reasoning_content", None):
            content = THINK_RE.sub("", content)
        return content.strip()
