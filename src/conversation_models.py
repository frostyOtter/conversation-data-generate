from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, Literal, List, Dict, Union
from datetime import datetime


class UserMetadata(BaseModel):
    user_id: Optional[str] = None
    language: Optional[str] = None
    region: Optional[str] = None
    device: Optional[str] = None


class Attachment(BaseModel):
    url: HttpUrl
    content_type: str
    attachment_type: Literal["image", "pdf", "file", "other"]


class UserMessage(BaseModel):
    message_id: str
    parent_id: Optional[str] = None
    text: str
    role: Literal["user"] = "user"
    attachments: Optional[List[Attachment]] = None
    timestamp: datetime


class ErrorDetail(BaseModel):
    code: Optional[str] = None
    message: str
    retryable: Optional[bool] = False


class LatencyStats(BaseModel):
    total_ms: int
    network_ms: Optional[int] = None
    inference_ms: Optional[int] = None
    postprocess_ms: Optional[int] = None


class ToolCallIO(BaseModel):
    function_tool: str
    input_params: Dict[str, Union[str, int, float, bool]]
    output_content: List[str]
    success: bool
    latency_ms: int
    error: Optional[ErrorDetail] = None


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Feedback(BaseModel):
    thumbs_up: Optional[bool] = None
    comment: Optional[str] = None


class TurnSummary(BaseModel):
    intent: Optional[str] = None
    tools_used: Optional[List[str]] = None
    completion_type: Optional[Literal["streamed", "function_call", "text_only"]] = None


class AssistantResponse(BaseModel):
    message_id: str
    parent_id: Optional[str] = None
    text: str
    role: Literal["assistant"] = "assistant"
    tool_calls: Optional[List[ToolCallIO]] = None
    assistant_success: bool
    function_call_success: Optional[bool] = None
    final_output_success: Optional[bool] = None
    latency: LatencyStats
    token_usage: Optional[TokenUsage] = None
    error: Optional[ErrorDetail] = None
    feedback: Optional[Feedback] = None
    generated_at: datetime
    received_at: datetime


class Turn(BaseModel):
    turn_id: int
    initiator_role: Literal["user", "assistant"]
    started_at: datetime
    user_message: Optional[UserMessage] = None
    assistant_response: Optional[AssistantResponse] = None
    summary: Optional[TurnSummary] = None


class ConversationSummary(BaseModel):
    total_turns: int
    average_processing_time_ms: int
    average_latency_ms: int


class Conversation(BaseModel):
    id: str
    language: str
    status: Literal["active", "completed", "failed"]
    turns: List[Turn]
    summary: ConversationSummary
    tags: Optional[List[str]] = []
    user_metadata: Optional[UserMetadata] = None
    schema_version: str = Field(default="2.0.0")
