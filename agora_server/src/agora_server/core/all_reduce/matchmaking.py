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

import asyncio
import re
import time

import numpy as np

from agora_server.hivemind.averaging.control import StepControl
from agora_server.hivemind.averaging.group_info import GroupInfo
from agora_server.hivemind.dht import DHT
from agora_server.hivemind.p2p import PeerID
from agora_server.hivemind.utils import DHTExpiration, ValueWithExpiration, get_dht_time, get_logger


GroupKey = str
Endpoint = str
GROUP_PATTERN = re.compile("^(([^.])+)[.]0b[01]*$")  # e.g. bert_exp4_averaging.0b01001101
logger = get_logger(__name__)


def is_valid_group(maybe_group: str) -> bool:
    """A group identifier must contain group type, followed by one or more .-separated indices, and any ?metadata."""
    return bool(GROUP_PATTERN.fullmatch(maybe_group))


class GroupKeyManager:
    """Utility class that manages per-step DHT group keys.

    Each global_step maps to a distinct DHT key so that responsible storage
    nodes rotate across the network, spreading load evenly.
    """

    def __init__(
        self,
        dht: DHT,
        prefix: str,
    ):
        self.dht = dht
        self.prefix = prefix
        self._key_prefix = prefix
        self.peer_id = dht.peer_id
        self.group_bits = ""

    def key_for_step(self, global_step: int) -> GroupKey:
        """Return a step-specific DHT key so each step hashes to different responsible nodes."""
        return f"{self._key_prefix}_s{global_step}.0b"

    async def declare_averager(
        self,
        peer_id: PeerID,
        expiration_time: float,
        global_step: int,
        data_for_gather: bytes | None = None,
        looking_for_group: bool = True,
    ) -> bool:
        """Add (or remove) the averager to the fixed group.

        Args:
            peer_id (PeerID): Averager public peer_id for incoming requests.
            expiration_time (float): Intent to run allreduce before this timestamp.
            global_step (int): The global training step this declaration is valid for.
            data_for_gather (bytes | None, optional): Optional data to share with the group. Defaults to None.
            looking_for_group (bool, optional): By default (True), declare the averager as "looking for group". If False, mark that the averager is no longer looking for group. Defaults to True.

        Returns:
            bool: True if declared, False if declaration was rejected by DHT peers.
        """
        expiration_time = expiration_time if looking_for_group else float(np.nextafter(expiration_time, float("inf")))
        return await self.dht.store(
            key=self.key_for_step(global_step),
            subkey=peer_id.to_bytes(),
            value=(looking_for_group, data_for_gather),
            expiration_time=expiration_time,
            return_future=True,
        )

    async def get_averagers(
        self,
        global_step: int,
        only_active: bool = True,
    ) -> list[tuple[PeerID, bytes, DHTExpiration]]:
        """Find and return averagers in the fixed group.

        Args:
            global_step (int): The global training step to get averagers for.
            only_active (bool, optional): If True, return only active averagers that are looking for group. If False, return all averagers in the group regardless of status. Defaults to True.

        Returns:
            list[tuple[PeerID, bytes, DHTExpiration]]: Peer_ids and expirations of every matching averager.
        """
        step_key = self.key_for_step(global_step)
        result = await self.dht.get(step_key, latest=True, return_future=True)
        if result is None or not isinstance(result.value, dict):
            logger.debug(f"No group found for step {global_step} (key={step_key})")
            return []

        averagers = []
        for key, res in result.value.items():
            looking_for_group, data_for_gather = res.value  # unpack the tuple
            try:
                if only_active and not looking_for_group:
                    continue
                averagers.append((PeerID(key), data_for_gather, res.expiration_time))
            except Exception as e:
                logger.warning(f"Could not parse peer key {key} ({looking_for_group}, exc={e})")
        return averagers

    async def join_group(self, expiration_time: float, global_step: int, data_for_gather: bytes) -> bool:
        """Join the fixed group (convenience method).

        Args:
            expiration_time (float): When this declaration expires.
            global_step (int): The global training step to join the group for.
            data_for_gather (bytes): Data to share with the group.

        Returns:
            bool: True if successfully joined.
        """
        return await self.declare_averager(
            self.peer_id, expiration_time, global_step, data_for_gather, looking_for_group=True
        )

    async def leave_group(self, expiration_time: float, global_step: int) -> bool:
        """Leave the fixed group (convenience method).

        Args:
            expiration_time (float): Original expiration time used when joining.

        Returns:
            bool: True if successfully left.
        """
        return await self.declare_averager(self.peer_id, expiration_time, global_step, looking_for_group=False)


class Matchmaking:
    """Simplified matchmaking that works with a single fixed group."""

    def __init__(
        self,
        dht: DHT,
        prefix: str,
        request_timeout: float = 5.0,  # TODO: remove this parameter
        min_matchmaking_time: float = 10.0,
        check_interval: float = 1.0,
        update_period: float = 3.0,  # TODO: remove this parameter
    ):
        """Initialize the matchmaking manager.

        Args:
            dht (DHT): An instance of hivemind.DHT. Server will use DHT for all network interactions.
            prefix (str): Prefix of the stage. i.e head, body0, tail.
            request_timeout (float, optional): This timeout is backward compatible with HM Matchmaking. It is used to cancel matchmaking in case of no responses. Defaults to 5.0.
            min_matchmaking_time (float, optional): How long to wait for peers to join the group. Defaults to 10.0.
            check_interval (float, optional): How often to poll the DHT for peers in the group. Defaults to 1.0.
            update_period (float, optional): How often to update the peer table with all peers in the stage. Defaults to 3.0. Deprecated.
        """
        self.group_key_manager = GroupKeyManager(dht, prefix)

        self.dht = dht
        self.prefix = prefix
        self.peer_id = self.group_key_manager.peer_id
        self.request_timeout = request_timeout
        self.min_matchmaking_time = min_matchmaking_time
        self.check_interval = check_interval

    def get_max_peers(self):
        try:
            response = self.dht.get(self.prefix.split("_")[0] + ".0.", latest=True)
            if isinstance(response, ValueWithExpiration) and isinstance(response.value, dict):
                return len(response.value)

            logger.warning(f"Incorrect peer response: {response}")
            return 0
        except Exception as e:
            logger.warning(f"Could not get max peers in the stage: {e}")
            return 0

    async def look_for_group(self, step: StepControl, global_step: int) -> GroupInfo | None:
        """Look for peers in the fixed group and form a group if enough peers are available.

        Args:
            step (StepControl): To get the step schedule time.
            global_step (int): The global training step to match peers for.

        Returns:
            GroupInfo | None: GroupInfo if group formed successfully, None otherwise.
        """
        # Announce that the averager is looking for group
        timeout = self.min_matchmaking_time

        new_expiration_time = float(get_dht_time() + timeout)
        await self.group_key_manager.join_group(new_expiration_time, global_step, step.data_for_gather)

        # Wait for peers to join the group
        start_time = time.time()
        max_peers = self.get_max_peers()

        # Accumulate all peers that issue join_group
        all_peers_data = {}  # peer_id_bytes -> data_for_gather
        while time.time() - start_time < timeout:
            averagers = await self.group_key_manager.get_averagers(global_step, only_active=True)
            for peer_id, data_for_gather, _ in averagers:
                all_peers_data[peer_id.to_bytes()] = data_for_gather

            if len(all_peers_data) > max_peers:
                # Outdated max_peers information, check again
                max_peers = self.get_max_peers()
                await asyncio.sleep(self.check_interval)
                continue

            if (len(all_peers_data) == max_peers) and (max_peers > 0):
                # We have enough peers, proceed with group formation
                break

            # Wait for either the peer_table to populate or to find all peers in the table
            logger.debug(f"Not enough peers yet: {len(all_peers_data)} < {max_peers}, waiting...")
            await asyncio.sleep(self.check_interval)

        if len(all_peers_data) == 0:
            # Timeout reached without finding enough peers
            logger.info("Timeout: Not any peers in group")
            return None

        # Create group info with all available peers
        all_peer_id_bytes = sorted(all_peers_data.keys())
        group_id = b"O[\x9aU\xcf%\xf0(\x90Nq\xdf!\x8b\x85)&\x0c\xe9r"
        peer_ids = [PeerID(peer_id_bytes) for peer_id_bytes in all_peer_id_bytes]
        gathered = tuple(all_peers_data[peer_id_bytes] for peer_id_bytes in all_peer_id_bytes)

        group_info = GroupInfo(group_id=group_id, peer_ids=tuple(peer_ids), gathered=gathered)

        end_time = time.time()
        logger.info(f"Formed group with {len(peer_ids)} peers out of {max_peers} in {end_time - start_time:.3f} secs")
        return group_info

    async def leave_group(self, expiration_time: float, global_step: int) -> bool:
        """Leave the current group.

        Args:
            expiration_time (float): Original expiration time.
            global_step (int): The global training step to leave the group for.

        Returns:
            bool: True if successfully left.
        """
        return await self.group_key_manager.leave_group(expiration_time, global_step)


class MatchmakingException(Exception):  # TODO: rename to MatchmakingError
    """An internal exception that marks undesired edge cases during averaging."""
