# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pydantic models for API requests and responses.
"""

from dataclasses import field
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    """Request model for the /generate/ endpoint."""
    prompts: list[str]
    images: list[str] | None = None
    n: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 16
    truncate_prompt_tokens: int | None = None
    guided_decoding_regex: str | None = None
    generation_kwargs: dict = field(default_factory=dict)


class GenerateResponse(BaseModel):
    """Response model for the /generate/ endpoint."""
    prompt_ids: list[list[int]]
    completion_ids: list[list[int]]
    logprobs: list[list[float]]


class ChatRequest(BaseModel):
    """Request model for the /chat/ endpoint."""
    messages: list[list[dict]]
    n: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: int = 16
    truncate_prompt_tokens: int | None = None
    guided_decoding_regex: str | None = None
    generation_kwargs: dict = field(default_factory=dict)
    chat_template_kwargs: dict = field(default_factory=dict)


class ChatResponse(BaseModel):
    """Response model for the /chat/ endpoint."""
    prompt_ids: list[list[int]]
    completion_ids: list[list[int]]
    logprobs: list[list[float]]


class InitCommunicatorRequest(BaseModel):
    """Request model for the /init_communicator/ endpoint."""
    host: str
    port: int
    world_size: int
    client_device_uuid: str


class UpdateWeightsRequest(BaseModel):
    """Request model for the /update_named_param/ endpoint."""
    name: str
    dtype: str
    shape: list[int]
