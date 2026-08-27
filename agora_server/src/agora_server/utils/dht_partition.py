# Copyright 2026 Pluralis Research
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


def partition_array(x: int, y: int) -> list[list[int]]:
    """Partition array [0, 1, 2, ..., x-1] into y partitions as equally as possible.

    Args:
        x (int): Size of array (values 0 to x-1).
        y (int): Number of partitions.

    Returns:
        list[list[int]]: List of lists representing the partitions.
    """
    if x == 0:
        return []

    if x <= y:
        # Each value gets its own partition
        return [[i] for i in range(y) if i < x]

    base_size = x // y
    remainder = x % y

    partitions = []
    start = 0

    for i in range(y):
        # First 'remainder' partitions get an extra element
        size = base_size + (1 if i < remainder else 0)
        partitions.append(list(range(start, start + size)))
        start += size

    return partitions


def stage_to_dht_map(dht_partition: list[list[int]]) -> list[int]:
    """Map stage index to dht index.

    Args:
        dht_partition (list[list[int]]): List of lists representing the partitions.

    Returns:
        list[int]: Stage to dht mapping.
    """
    stage_to_dht = [part for part, sublist in enumerate(dht_partition) for _ in sublist]
    return stage_to_dht


def update_initial_peers(
    initial_peers: list[str],
    pipeline_stage: str,
    num_stages: int,
    num_dht: int,
) -> list[str]:
    """Update the list of initial peers with correct ports that match the given stage.

    Args:
        initial_peers (list[str]): List of multiaddress.
        pipeline_stage (str): Stage type in the format: head-X, body-X, tail-X (X is int).
        num_stages (int): Total number of stages.
        num_dht (int): Number of worker DHTs.

    Raises:
        ValueError: Wwrong stage type.

    Returns:
        list[str]: Updated `initial_peers`.
    """

    # Calculate port offset according to stage
    stage_idx = int(pipeline_stage.split("-")[1])
    dht_worker_partitions = partition_array(num_stages, num_dht)
    stage_to_dht = stage_to_dht_map(dht_worker_partitions)
    port_offset = stage_to_dht[stage_idx]

    # Update initial peers ports
    for i, peeri in enumerate(initial_peers):
        try:
            # Extract baseline port
            parts = peeri.split("/")
            port_index = parts.index("tcp") + 1
            base_port = int(parts[port_index])
            parts[port_index] = str(base_port + port_offset)
            initial_peers[i] = "/".join(parts)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid multiaddress format in peer {i}: {peeri}. Error: {e}") from e

    return initial_peers
