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
vLLM worker extension for weight synchronization.
"""

from collections.abc import Sequence

import torch

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import StatelessProcessGroup


class WeightSyncWorkerExtension:
    """
    A vLLM worker extension that enables weight synchronization between a client and multiple server workers.

    This worker uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` or
    `ProcessGroupXCCL` to handle efficient GPU-based communication using NCCL. The primary purpose of this class is to
    receive updated model weights from a client process and distribute them to all worker processes participating in
    model inference.
    """

    # The following attributes are initialized when `init_communicator` method is called.
    communicator = None  # Communicator for weight updates
    client_rank = None  # Source rank for broadcasting updated weights

    def init_communicator(self, host: str, port: int, world_size: int, client_device_uuid: str) -> None:
        """
        Initializes the weight update communicator using a stateless process group.

        This method creates a `StatelessProcessGroup` that allows external training processes to communicate with vLLM
        workers without interfering with the global torch distributed group.

        Args:
            host (`str`):
                Hostname or IP address of the master node.
            port (`int`):
                Port number to be used for communication.
            world_size (`int`):
                Total number of participating processes in the update group.
            client_device_uuid (`str`):
                UUID of the device of client main process. Used to assert that devices are different from vllm workers devices.
        """
        if self.communicator is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        if torch.cuda.is_available():
            if client_device_uuid == str(torch.cuda.get_device_properties(self.device).uuid):
                raise RuntimeError(
                    f"Attempting to use the same CUDA device (UUID: {client_device_uuid}) for multiple distinct "
                    "roles/ranks within the same communicator. This setup is unsupported and will likely lead to program "
                    "hangs or incorrect behavior. Ensure that trainer is using different devices than vLLM server."
                )
        # Get the rank of the current worker in the global world group.
        rank = get_world_group().rank

        # Create a stateless process group to manage communication between training processes and vLLM workers.
        # Initialize the NCCL-based communicator for weight synchronization.
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        self.communicator = PyNcclCommunicator(pg, device=self.device)

        # The client process that sends updated weights has the highest rank (world_size - 1).
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`str`):
                Data type of the weight tensor as a string (e.g., `"torch.float32"`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """
        if self.communicator is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        dtype = getattr(torch, dtype.split(".")[-1])
        # Allocate memory for the incoming weight tensor on the correct device.
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        # Use NCCL to broadcast the updated weights from the client (src) to all workers.
        self.communicator.broadcast(weight, src=self.client_rank)
        self.communicator.group.barrier()

        # Load the received weights into the model.
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        """
        Closes the communicator when weight synchronization is no longer needed.

        This method deletes the NCCL communicator to release associated resources.
        """

        if self.communicator is not None:
            del self.communicator
            self.communicator = None  # Ensure attribute is reset to None
            self.client_rank = None  # Ensure attribute is reset to None
