# src/llm_client.py
from __future__ import annotations

import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai import (
    BadRequestError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from pydantic import BaseModel, Field

load_dotenv()


class Label12(BaseModel):
    # 12 nhãn (dùng bool để schema đơn giản, ổn định)
    F: bool
    A: bool
    FT: bool
    L: bool
    LF: bool
    MN: bool
    O: bool
    PE: bool
    PO: bool
    SC: bool
    SE: bool
    US: bool

    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str


RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


class LabelingAgent:
    def __init__(self, model: str, reasoning_effort: Optional[str] = "low"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Missing OPENAI_API_KEY. Put it in .env (or system env).")

        self.model = model
        self.reasoning_effort = reasoning_effort

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=20),
        retry=retry_if_exception_type(RETRYABLE),
    )
    def _call(self, system_prompt: str, user_prompt: str) -> Label12:
        kwargs = {}
        if self.reasoning_effort:
            # GPT-5 reasoning effort thường: low/medium/high (một số model mới có none)
            kwargs["reasoning"] = {"effort": self.reasoning_effort}

        resp = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=Label12,
            **kwargs,
        )
        return resp.output_parsed

    def classify(
        self,
        requirement_text: str,
        system_prompt: str,
        examples: Optional[List[Dict]] = None,
    ) -> Label12:
        ex_block = ""
        if examples:
            # examples: [{"text": "...", "labels": {"F":1,...}}]
            lines = []
            for ex in examples:
                lab = ex.get("labels", {})
                lines.append(
                    f'- Text: "{ex.get("text","")}"\n'
                    f"  Labels: {lab}"
                )
            ex_block = "\n\nSimilar labeled examples:\n" + "\n\n".join(lines)

        user_prompt = (
            f'RequirementText:\n"{requirement_text}"\n'
            f"{ex_block}\n\n"
            "Return ONLY the JSON that matches the schema."
        )

        try:
            return self._call(system_prompt, user_prompt)
        except BadRequestError as e:
            # Quan trọng: in lỗi 400 thật để biết sai ở đâu
            msg = getattr(e, "message", None) or str(e)
            raise BadRequestError(msg, response=getattr(e, "response", None), body=getattr(e, "body", None))
