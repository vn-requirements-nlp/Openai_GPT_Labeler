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
    # Bắt buộc model phải suy luận trước
    # Đây là kỹ thuật quan trọng để tăng độ chính xác cho model nhỏ.
    rationale: str = Field(..., description="A short analysis (1-3 sentences). 1. Identify keywords. 2. Explain why specific labels are selected or rejected based on the definitions.")
    
    # Sau khi đã suy luận, mới chốt kết quả
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
        
        # 1. Context Block (RAG)
        context_block = ""
        if examples:
            ex_lines = []
            for i, ex in enumerate(examples, 1):
                # Chỉ lấy các label True để làm ví dụ cho gọn
                true_labels = [k for k, v in ex.get("labels", {}).items() if v is True]
                ex_lines.append(f"Example {i}:\n- Input: \"{ex.get('text','')}\"\n- Classified as: {', '.join(true_labels) if true_labels else 'None'}")
            
            context_block = (
                "### REFERENCE EXAMPLES (Use these to understand the labeling style, but do not copy blindly)\n"
                f"{chr(10).join(ex_lines)}\n\n"
            )

        # 2. User Prompt được cấu trúc rõ ràng với Delimiters
        user_prompt = (
            f"{context_block}"
            "### TARGET REQUIREMENT\n"
            f"Input: \"{requirement_text}\"\n\n"
            "### INSTRUCTION\n"
            "Analyze the Target Requirement above based on the system definitions. "
            "Think step-by-step in the 'rationale' field before setting labels."
        )

        try:
            return self._call(system_prompt, user_prompt)
        except BadRequestError as e:
            # Quan trọng: in lỗi 400 thật để biết sai ở đâu
            msg = getattr(e, "message", None) or str(e)
            raise BadRequestError(msg, response=getattr(e, "response", None), body=getattr(e, "body", None))
