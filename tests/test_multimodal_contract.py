"""Contract tests for multimodal chat input support."""

import pytest

from infer_nexus.backends.vllm import VLLMBackend
from infer_nexus.core.errors import BackendRequestValidationError
from infer_nexus.core.schemas import ChatCompletionsRequest


TEXT_AND_IMAGE_DATA_URL_REQUEST = ChatCompletionsRequest(
    model="qwen3-vl-chat-8b-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
                    },
                },
            ],
        }
    ],
)


def test_chat_message_schema_accepts_multimodal_content_blocks() -> None:
    """Schema should continue accepting OpenAI-style multimodal blocks."""
    request = TEXT_AND_IMAGE_DATA_URL_REQUEST

    assert request.messages[0].role == "user"
    assert isinstance(request.messages[0].content, list)
    assert request.messages[0].content[0].type == "text"
    assert request.messages[0].content[1].type == "image_url"


def test_vllm_backend_should_accept_text_and_image_content_blocks() -> None:
    """Backend should pass multimodal content through once support is implemented."""
    backend = VLLMBackend({"capabilities": ["vision"]})

    messages = backend._build_chat_messages(TEXT_AND_IMAGE_DATA_URL_REQUEST)

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
                    },
                },
            ],
        }
    ]


def test_vllm_backend_should_accept_remote_image_urls() -> None:
    """Backend should accept HTTPS image URLs for vision-capable models."""
    backend = VLLMBackend({"capabilities": ["vision"]})
    request = ChatCompletionsRequest(
        model="qwen3-vl-chat-8b-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in the picture?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/demo.png"},
                    },
                ],
            }
        ],
    )

    messages = backend._build_chat_messages(request)

    assert messages[0]["content"][1]["image_url"]["url"] == "https://example.com/demo.png"


def test_vllm_backend_normalizes_data_url_payload_before_passing_to_vllm() -> None:
    """Whitespace and URL-safe base64 should be normalized for vLLM."""
    backend = VLLMBackend({"capabilities": ["vision"]})
    request = ChatCompletionsRequest(
        model="mineru",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aGV sbG8_\n"},
                    },
                ],
            }
        ],
    )

    messages = backend._build_chat_messages(request)

    assert messages[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,aGVsbG8/="


def test_vllm_backend_rejects_multimodal_content_for_non_vision_models() -> None:
    """Non-vision models should still reject image blocks."""
    backend = VLLMBackend({})

    with pytest.raises(
        BackendRequestValidationError,
        match="does not support multimodal",
    ):
        backend._build_chat_messages(TEXT_AND_IMAGE_DATA_URL_REQUEST)
