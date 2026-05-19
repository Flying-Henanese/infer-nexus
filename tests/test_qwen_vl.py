from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from openai import OpenAI

API_BASE = "http://127.0.0.1:8000/v1"
MODEL = "qwen3-vl-chat-8b-instruct"

# TODO: 填你自己的图片路径
IMAGE_PATH = "test.jpg"


def to_data_url(image_path: str) -> str:
    path = Path(image_path).expanduser().resolve()
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/png"

    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def main() -> None:
    if IMAGE_PATH == "YOUR_IMAGE_PATH_HERE":
        raise ValueError("请先把 IMAGE_PATH 改成真实图片路径")

    client = OpenAI(base_url=API_BASE, api_key="dummy")

    image_data_url = to_data_url(IMAGE_PATH)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这张图片的主要内容。"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                ],
            }
        ],
        max_tokens=512,
    )

    print(resp.model)
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()