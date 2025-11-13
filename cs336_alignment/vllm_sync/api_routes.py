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
API route handlers for the vLLM server.
"""

import base64
import logging
import math
from io import BytesIO
from itertools import chain

from fastapi import APIRouter
from PIL import Image

from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from .api_models import (
    GenerateRequest,
    GenerateResponse,
    ChatRequest,
    ChatResponse,
    InitCommunicatorRequest,
    UpdateWeightsRequest,
)
from .config import ScriptArguments
from .worker import chunk_list

logger = logging.getLogger(__name__)


def sanitize_logprob(logprob):
    """
    Sanitizes log probabilities to handle NaN values.

    Args:
        logprob: The log probability object to sanitize.

    Returns:
        The log probability value if valid, None if NaN.
    """
    value = logprob.logprob
    if math.isnan(value):
        logger.warning(f"Generated NaN logprob, token logprob '{logprob}' will be ignored")
        return None
    return value


def create_router(script_args: ScriptArguments, connections: list) -> APIRouter:
    """
    Creates and configures the API router with all endpoints.

    Args:
        script_args (`ScriptArguments`):
            Configuration arguments for the server.
        connections (list):
            List of pipe connections to worker processes.

    Returns:
        `APIRouter`: Configured router with all endpoints.
    """
    router = APIRouter()

    @router.get("/health/")
    async def health():
        """
        Health check endpoint to verify that the server is running.
        """
        return {"status": "ok"}

    @router.get("/get_world_size/")
    async def get_world_size():
        """
        Retrieves the world size of the LLM engine, which is `tensor_parallel_size * data_parallel_size`.

        Returns:
            `dict`:
                A dictionary containing the world size.

        Example response:
        ```json
        {"world_size": 8}
        ```
        """
        return {"world_size": script_args.tensor_parallel_size * script_args.data_parallel_size}

    @router.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Generates completions for the provided prompts.

        Args:
            request (`GenerateRequest`):
                - `prompts` (list of `str`): A list of prompts (text strings) for the model to generate completions.
                - `images` (list of `str`, *optional*, default to `None`): A list of base64 encoded images to process
                  along with prompts.
                - `n` (`int`, *optional*, defaults to `1`): Number of completions to generate for each prompt.
                - `repetition_penalty` (`float`, *optional*, defaults to `1.0`): Repetition penalty to apply during
                  generation.
                - `temperature` (`float`, *optional*, defaults to `1.0`): Temperature for sampling. Higher values lead
                  to more random outputs.
                - `top_p` (`float`, *optional*, defaults to `1.0`): Top-p (nucleus) sampling parameter. It controls the
                  diversity of the generated text.
                - `top_k` (`int`, *optional*, defaults to `-1`): Top-k sampling parameter. If set to `-1`, it disables
                  top-k sampling.
                - `min_p` (`float`, *optional*, defaults to `0.0`): Minimum probability threshold for sampling.
                - `max_tokens` (`int`, *optional*, defaults to `16`): Maximum number of tokens to generate for each
                  completion.
                - `truncate_prompt_tokens` (`int`, *optional*): If set to `-1`, will use the truncation size supported
                  by the model. If set to an integer k, will use only the last k tokens from the prompt (i.e., left
                  truncation). If set to `None`, truncation is disabled.
                - `guided_decoding_regex` (`str`, *optional*): A regex pattern for guided decoding. If provided, the
                  model will only generate tokens that match this regex pattern.
                - `generation_kwargs` (`dict`, *optional*): Additional generation parameters to pass to the vLLM
                  `SamplingParams`. This can include parameters like `seed`, `frequency_penalty`, etc. If it contains
                  keys that conflict with the other parameters, they will override them.

        Returns:
            `GenerateResponse`:
                - `prompt_ids` (list of list of `int`): A list of lists of token IDs for each input prompt.
                - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.
                - `logprobs` (list of list of `float`): A list of lists of log probabilities for each token in the
                  generated completions.

        Example request:
        ```json
        {"prompts": ["Hello world", "What is AI?"]}
        ```

        Example response:
        ```json
        {
          "prompt_ids": [[101, 102], [201, 202]],
          "completion_ids": [[103, 104, 105], [203, 204, 205]],
          "logprobs": [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]]
        }
        ```
        """
        request.images = request.images or [None] * len(request.prompts)

        prompts = []
        for prompt, image in zip(request.prompts, request.images, strict=True):
            row = {"prompt": prompt}
            if image is not None:
                row["multi_modal_data"] = {"image": Image.open(BytesIO(base64.b64decode(image)))}
            prompts.append(row)

        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(regex=request.guided_decoding_regex)
        else:
            guided_decoding = None

        generation_kwargs = {
            "n": request.n,
            "repetition_penalty": request.repetition_penalty,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "max_tokens": request.max_tokens,
            "truncate_prompt_tokens": request.truncate_prompt_tokens,
            "guided_decoding": guided_decoding,
            "logprobs": 0,  # enable returning log probabilities; 0 means for the sampled tokens only
        }
        generation_kwargs.update(request.generation_kwargs)
        sampling_params = SamplingParams(**generation_kwargs)

        # Evenly distribute prompts across DP ranks
        chunked_prompts = chunk_list(prompts, script_args.data_parallel_size)

        # Send the prompts to each worker
        for connection, prompts in zip(connections, chunked_prompts, strict=True):
            # When the number of prompts is less than data_parallel_size, some workers will receive empty prompts.
            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply
            # with vLLM's requirement, and we later ignore the result.
            if not prompts:
                prompts = ["<placeholder>"]
            kwargs = {"prompts": prompts, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})

        # Receive results
        all_outputs = [connection.recv() for connection in connections]

        # Handle empty prompts (see above)
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_prompts, strict=True) if prompts]

        # Flatten and combine all results
        all_outputs = list(chain.from_iterable(all_outputs))  # from list of list to single list
        prompt_ids = [output.prompt_token_ids for output in all_outputs]
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        logprobs: list[list[float]] = [
            [sanitize_logprob(next(iter(logprob.values()))) for logprob in output.logprobs]
            for outputs in all_outputs
            for output in outputs.outputs
        ]
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": logprobs}

    @router.post("/chat/", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Generates completions for the provided chat messages.

        Args:
            request (`ChatRequest`):
                - `messages` (list of `dict`): A list of messages (dicts with "role" and "content" keys) for the model
                  to generate completions.
                - `n` (`int`, *optional*, defaults to `1`): Number of completions to generate for each prompt.
                - `repetition_penalty` (`float`, *optional*, defaults to `1.0`): Repetition penalty to apply during
                  generation.
                - `temperature` (`float`, *optional*, defaults to `1.0`): Temperature for sampling. Higher values lead
                  to more random outputs.
                - `top_p` (`float`, *optional*, defaults to `1.0`): Top-p (nucleus) sampling parameter. It controls the
                  diversity of the generated text.
                - `top_k` (`int`, *optional*, defaults to `-1`): Top-k sampling parameter. If set to `-1`, it disables
                  top-k sampling.
                - `min_p` (`float`, *optional*, defaults to `0.0`): Minimum probability threshold for sampling.
                - `max_tokens` (`int`, *optional*, defaults to `16`): Maximum number of tokens to generate for each
                  completion.
                - `truncate_prompt_tokens` (`int`, *optional*): If set to `-1`, will use the truncation size supported
                  by the model. If set to an integer k, will use only the last k tokens from the prompt (i.e., left
                  truncation). If set to `None`, truncation is disabled.
                - `guided_decoding_regex` (`str`, *optional*): A regex pattern for guided decoding. If provided, the
                  model will only generate tokens that match this regex pattern.
                - `generation_kwargs` (`dict`, *optional*): Additional generation parameters to pass to the vLLM
                  `SamplingParams`. This can include parameters like `seed`, `frequency_penalty`, etc. If it contains
                  keys that conflict with the other parameters, they will override them.
                - `chat_template_kwargs` (`dict`, *optional*): Additional keyword arguments to pass to the chat
                  template.

        Returns:
            `ChatResponse`:
                - `prompt_ids` (list of list of `int`): A list of lists of token IDs for each input prompt.
                - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.
                - `logprobs` (list of list of `float`): A list of lists of log probabilities for each token in the
                  generated completions.

        Example request:
        ```bash
        curl -X POST 'http://0.0.0.0:8000/chat/' \
          -H 'Content-Type: application/json' \
          -d '{"messages": [[{ "role": "user", "content": "Hello!" }]]}'
        ```

        Example response:
        ```json
        {
            "prompt_ids": [[151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198]],
            "completion_ids":[[151667, 198, 32313, 11, 279, 1196, 1101, 1053, 330, 9707, 8958, 773, 358, 1184, 311, 5889]],
            "logprobs": [[-0.00029404606902971864, -3.576278118089249e-07, -0.09024181962013245, -6.389413465512916e-05, -0.038671817630529404, -0.00013314791431184858, -0.5868351459503174, -0.09682723134756088, -0.06609706580638885, -0.00023803261865396053, -0.02242819033563137, -0.8185162544250488, -0.04954879730939865, -0.3169460594654083, -4.887569048150908e-06, -0.006023705471307039]]
        }
        ```
        """
        # Convert PIL images to base64 strings
        for message_list in request.messages:
            for message in message_list:
                if isinstance(message["content"], list):
                    for part in message["content"]:
                        if part["type"] == "image_pil":
                            part["image_pil"] = Image.open(BytesIO(base64.b64decode(part["image_pil"])))

        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(regex=request.guided_decoding_regex)
        else:
            guided_decoding = None

        generation_kwargs = {
            "n": request.n,
            "repetition_penalty": request.repetition_penalty,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "max_tokens": request.max_tokens,
            "truncate_prompt_tokens": request.truncate_prompt_tokens,
            "guided_decoding": guided_decoding,
            "logprobs": 0,  # enable returning log probabilities; 0 means for the sampled tokens only
        }
        generation_kwargs.update(request.generation_kwargs)
        sampling_params = SamplingParams(**generation_kwargs)

        # Evenly distribute prompts across DP ranks
        chunked_messages = chunk_list(request.messages, script_args.data_parallel_size)

        # Send the messages to each worker
        for connection, messages in zip(connections, chunked_messages, strict=True):
            # When the number of messages is less than data_parallel_size, some workers will receive empty messages.
            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply
            # with vLLM's requirement, and we later ignore the result.
            if not messages:
                messages = [[{"role": "user", "content": "<placeholder>"}]]
            kwargs = {
                "messages": messages,
                "sampling_params": sampling_params,
                "chat_template_kwargs": request.chat_template_kwargs,
            }
            connection.send({"type": "call", "method": "chat", "kwargs": kwargs})

        # Receive results
        all_outputs = [connection.recv() for connection in connections]

        # Handle empty prompts (see above)
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_messages, strict=True) if prompts]

        # Flatten and combine all results
        all_outputs = list(chain.from_iterable(all_outputs))  # from list of list to single list
        prompt_ids = [output.prompt_token_ids for output in all_outputs]
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        logprobs: list[list[float]] = [
            [sanitize_logprob(next(iter(logprob.values()))) for logprob in output.logprobs]
            for outputs in all_outputs
            for output in outputs.outputs
        ]
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": logprobs}

    @router.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        """
        Initializes the communicator for synchronizing model weights between a client and multiple server workers.

        Args:
            request (`InitCommunicatorRequest`):
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the group.
                - `client_device_uuid` (`str`): UUID of the device of client main process. Used to assert that devices
                  are different from vLLM workers devices.
        """
        world_size = script_args.tensor_parallel_size * script_args.data_parallel_size + 1

        # The function init_communicator is called this way: init_communicator(host, port, world_size)
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc(method="init_communicator", args=(host, port, world_size))
        kwargs = {
            "method": "init_communicator",
            "args": (request.host, request.port, world_size, request.client_device_uuid),
        }
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})

        return {"message": "Request received, initializing communicator"}

    @router.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
        # The function update_named_param is called this way: update_named_param("name", "torch.float32", (10, 10))
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc("update_named_param", args=("name", "torch.float32", (10, 10)))
        kwargs = {"method": "update_named_param", "args": (request.name, request.dtype, tuple(request.shape))}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})

        return {"message": "Request received, updating named parameter"}

    @router.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        """
        Resets the prefix cache for the model.
        """
        for connection in connections:
            connection.send({"type": "call", "method": "reset_prefix_cache"})
        # Wait for and collect all results
        all_outputs = [connection.recv() for connection in connections]
        success = all(output for output in all_outputs)
        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @router.post("/close_communicator/")
    async def close_communicator():
        """
        Closes the weight update group and cleans up associated resources.
        """
        kwargs = {"method": "close_communicator"}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "Request received, closing communicator"}

    return router
