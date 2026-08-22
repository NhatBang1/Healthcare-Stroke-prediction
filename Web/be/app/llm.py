from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings


SYSTEM_PROMPT = """Bạn là trợ lý AI y tế chuyên về đột quỵ, dùng cho giáo dục sức khỏe và sàng lọc nguy cơ. Bạn không phải bác sĩ và không được chẩn đoán chắc chắn.

Luật bắt buộc:
- Trả lời ngắn gọn, đúng trọng tâm, bằng tiếng Việt tự nhiên.
- Không tự bịa chỉ số, xác suất, nguồn tài liệu hoặc guideline.
- Không khẳng định người dùng chắc chắn bị hoặc không bị đột quỵ.
- Nếu người dùng mô tả méo miệng, yếu/tê một bên, nói khó, nhìn mờ đột ngột, đau đầu dữ dội hoặc mất thăng bằng đột ngột, hãy ưu tiên khuyến nghị gọi cấp cứu ngay.
- Khi hỏi về nguy cơ cá nhân, nhắc người dùng cung cấp: tuổi, giới tính, tăng huyết áp, bệnh tim, glucose trung bình và BMI.
- Kết quả sàng lọc nguy cơ không thay thế bác sĩ.
"""


class DeepSeekClient:
    def __init__(self) -> None:
        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_base_url.rstrip("/")
        self.model = settings.deepseek_model
        self.timeout = settings.deepseek_timeout_seconds

    def configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            "temperature": 0.2,
            "max_tokens": 700,
            "stream": False,
            "thinking": {"type": "disabled"},
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()


deepseek_client = DeepSeekClient()
