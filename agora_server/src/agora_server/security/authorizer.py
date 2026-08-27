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

import logging
import multiprocessing as mp
import os
import signal
import time

from datetime import datetime, timedelta, timezone

import requests

from tenacity import before_sleep_log, retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from agora_server.hivemind.proto.auth_pb2 import AccessToken
from agora_server.hivemind.utils.auth import TokenAuthorizerBase
from agora_server.hivemind.utils.crypto import RSAPrivateKey, RSAPublicKey
from agora_server.hivemind.utils.logging import get_logger
from agora_server.utils.node_info import NodeInfo


logger = get_logger(__name__)


class NonRetriableError(Exception):
    pass


class PluralisAuthorizer(TokenAuthorizerBase):
    def __init__(
        self,
        peer_id: str,
        user_token: str,
        role: str,
        auth_server: str,
        node_info: NodeInfo | None = None,
        expert_uid: str | None = None,
        hostname: str | None = None,
        local_private_key: RSAPrivateKey | None = None,
    ):
        super().__init__(local_private_key)

        self.peer_id = peer_id
        self._user_token = user_token
        self._role = role
        self._auth_server = auth_server.rstrip("/")
        self._node_info = node_info
        self.expert_uid = expert_uid
        self._hostname = hostname
        self._authority_public_key = None
        self._shutdown_event = mp.Event()

        # Populated from the server response once access is granted
        self.exported_role: str | None = None
        self.pithos_shard_index: int | None = None
        self.pithos_shard_count: int | None = None
        self.pithos_shards_per_head: int | None = None
        self.pithos_corpus_uri: str | None = None

        # Save request parameters for future use
        self.auth_header = {"Authorization": f"Bearer {self._user_token}"}
        self.peer_info = {
            "peer_id": self.peer_id,
            "role": self._role,
            "peer_public_key": self.local_public_key.to_bytes().decode(),
            "device": self._node_info.device_name if self._node_info else None,
            "gpu_memory": self._node_info.gpu_memory if self._node_info else None,
            "ram": self._node_info.ram if self._node_info else None,
            "download_speed": self._node_info.download_speed if self._node_info else None,
            "upload_speed": self._node_info.upload_speed if self._node_info else None,
            "latency": self._node_info.latency if self._node_info else None,
            "cloud_provider": self._node_info.cloud_provider if self._node_info else "unknown",
            "uid": self.expert_uid,
            "hostname": self._hostname,
        }

    async def get_token(self) -> AccessToken:
        """Hivemind calls this method to refresh the access token when necessary."""
        try:
            if not self._shutdown_event.is_set():
                self.refresh_token()
            return self._local_access_token
        except Exception as e:
            if not self._shutdown_event.is_set():
                logger.error(f"Authorization failed: {e}. Exiting run.")
                os.killpg(os.getpgrp(), signal.SIGINT)

    def extract_token(self, response: dict) -> AccessToken:
        """Extract access token from response json."""
        access_token = AccessToken()
        access_token.username = response["username"]
        access_token.public_key = response["peer_public_key"].encode()
        access_token.expiration_time = str(datetime.fromisoformat(response["expiration_time"]))
        access_token.signature = response["signature"].encode()
        # Role+stage claim ("role[:stage]"). Only present once the
        # auth_server's TOKEN_ROLE_CLAIM flag is on; when set, the token
        # signature covers it (see _token_to_bytes).
        access_token.role = response.get("role") or ""
        return access_token

    def _read_shard_assignment(self, response: dict) -> None:
        """Adopt the authorizer's data-shard assignment for head workers, when present."""
        self.pithos_shard_index = response.get("pithos_shard_index")
        self.pithos_shard_count = response.get("pithos_shard_count")
        self.pithos_shards_per_head = response.get("pithos_shards_per_head")
        self.pithos_corpus_uri = response.get("pithos_corpus_uri")

    @retry(
        retry=retry_if_not_exception_type(NonRetriableError),
        wait=wait_exponential(multiplier=2),
        stop=stop_after_attempt(7),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    def join_experiment(self) -> None:
        """Send authorization request to join the experiment and receive access token."""
        try:
            url = f"{self._auth_server}/api/join"
            response = requests.put(
                url,
                headers=self.auth_header,
                # Stamp a FRESH local timestamp per request (peer_info is built once at
                # init; the node may sit queued for a while) for the authorizer's
                # clock-skew check.
                json={**self.peer_info, "client_time": time.time()},
            )

            response.raise_for_status()
            response = response.json()

            self._authority_public_key = RSAPublicKey.from_bytes(response["auth_server_public_key"].encode())
            self._local_access_token = self.extract_token(response)
            self.exported_role = response.get("exported_role")
            self._read_shard_assignment(response)

            logger.info(
                f"Access for user {self._local_access_token.username} has been granted until {self._local_access_token.expiration_time} UTC"
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code != 503:
                try:
                    error_detail = e.response.json()["detail"]
                except Exception:
                    # AWS might return non-JSON error responses, so fallback to a generic message
                    error_detail = "Request is blocked from AWS"
                raise NonRetriableError(error_detail) from None
            raise e
        except requests.exceptions.ConnectionError as e:
            raise e
        except Exception as e:
            raise NonRetriableError(str(e)) from None

    @retry(
        retry=retry_if_not_exception_type(NonRetriableError),
        wait=wait_exponential(multiplier=2),
        stop=stop_after_attempt(7),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.INFO),
    )
    def refresh_token(self) -> None:
        """Refresh access token."""
        # Check if shutdown is in progress before attempting refresh
        if self._shutdown_event.is_set():
            raise NonRetriableError("Shutdown in progress, aborting token refresh")

        try:
            url = f"{self._auth_server}/api/refresh_token"
            response = requests.put(
                url,
                headers=self.auth_header,
                # Stamp a FRESH local timestamp per request (peer_info is built once at
                # init; the node may sit queued for a while) for the authorizer's
                # clock-skew check.
                json={**self.peer_info, "client_time": time.time()},
            )

            response.raise_for_status()
            response = response.json()
            self._local_access_token = self.extract_token(response)

            logger.info(
                f"Access for user {self._local_access_token.username} has been granted until {self._local_access_token.expiration_time} UTC"
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code != 503:
                try:
                    error_detail = e.response.json()["detail"]
                except Exception:
                    # AWS might return non-JSON error responses, so fallback to a generic message
                    error_detail = "Request is blocked from AWS"
                raise NonRetriableError(error_detail) from None
            raise e
        except requests.exceptions.ConnectionError as e:
            raise e
        except Exception as e:
            raise NonRetriableError(str(e)) from None

    def is_token_valid(self, access_token: AccessToken) -> bool:
        """Verify that token is valid."""
        data = self._token_to_bytes(access_token)
        if not self._authority_public_key.verify(data, access_token.signature):
            logger.exception("Access token has invalid signature")
            return False

        try:
            expiration_time = datetime.fromisoformat(access_token.expiration_time)
        except ValueError:
            logger.exception(
                f"datetime.fromisoformat() failed to parse expiration time: {access_token.expiration_time}"
            )
            return False
        if expiration_time < datetime.now(timezone.utc):
            logger.exception("Access token has expired")
            return False

        return True

    def begin_shutdown(self) -> None:
        """Mark that shutdown is in progress."""
        self._shutdown_event.set()

    _MAX_LATENCY = timedelta(minutes=1)

    def does_token_need_refreshing(self, access_token: AccessToken) -> bool:
        """Check if token has expired."""
        expiration_time = datetime.fromisoformat(access_token.expiration_time)
        return expiration_time < datetime.now(timezone.utc) + self._MAX_LATENCY

    @staticmethod
    def _token_to_bytes(access_token: AccessToken) -> bytes:
        """The byte string the auth_server's token signature covers.

        Include-role-iff-set: legacy tokens (empty ``role``) keep the
        original format, tokens carrying the signed role claim append it.
        Must stay byte-identical to the auth_server's
        ``grant_access`` and the Metadata Store facade's rebuild — including
        the historical quirk that ``public_key`` is a bytes object rendered
        through an f-string (``b'ssh-rsa ...'``).
        """
        data = f"{access_token.username} {access_token.public_key} {access_token.expiration_time}"
        if access_token.role:
            data += f" {access_token.role}"
        return data.encode()
