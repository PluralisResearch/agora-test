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

import threading
import time

from agora_server.hivemind.dht import DHT
from agora_server.hivemind.utils import ValueWithExpiration, get_logger


logger = get_logger(__name__)


class PeerVisibilityMonitor(threading.Thread):
    """Monitor peer visibility in the DHT within a stage."""

    def __init__(self, dht: DHT, uids: list, update_period: float = 180):
        super().__init__(daemon=True)

        self.dht = dht
        self.update_period = update_period
        self.keys = [f"{key}0." for key in uids]
        self.start()

    def _check_peer_visibility(self, key: str):
        response = self.dht.get(key, latest=True)
        if isinstance(response, ValueWithExpiration) and isinstance(response.value, dict):
            num_items = len(response.value.items())
            logger.info(f"Node can see {num_items} of {key}")
        else:
            logger.info(f"Node can see 0 of {key}")

    def run(self):
        while True:
            [self._check_peer_visibility(key) for key in self.keys]
            time.sleep(self.update_period)
