# Copyright 2026 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""AccessToken -> Metadata Store bearer plumbing.

The Metadata Store facade authenticates requests with the peer's
auth_server-issued ``AccessToken``, carried as
``Authorization: Bearer <base64url(msgpack(fields))>`` — the token signature
covers the *fields*, not the proto serialization, so the facade never parses
protobuf.

Tokens are short-lived (minutes for volunteers), so a static header string is
useless: :class:`MetadataStoreTokenProvider` wraps the authorizer and is called
per request by ``MetadataStoreClient`` — it serializes the current token to a
bearer and refreshes through the authorizer's (synchronous, retried)
``refresh_token`` when the token nears expiry. Fork-aware: the provider is carried into forked
consumers (averager children, the resolver's per-process readers), so its lock
is re-initialized after fork like the client's own locks.
"""

import base64
import multiprocessing.util as mp_util
import threading

from typing import Any, cast

import msgpack

from agora_server.hivemind.proto.auth_pb2 import AccessToken
from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)


def access_token_to_bearer(access_token: AccessToken) -> str:
    """Serialize an AccessToken into the facade's bearer value."""
    fields: dict[str, Any] = {
        "username": access_token.username,
        "public_key": access_token.public_key,
        "expiration_time": access_token.expiration_time,
        "role": access_token.role,
        "signature": access_token.signature,
    }
    packed = cast(bytes, msgpack.dumps(fields, use_bin_type=True))
    return base64.urlsafe_b64encode(packed).decode()


class MetadataStoreTokenProvider:
    """Callable returning the current bearer string for Metadata Store requests.

    Refresh policy mirrors hivemind's ``refresh_token_if_needed``: refresh when
    the authorizer reports the token stale (its ``_MAX_LATENCY`` margin before
    expiry), under a lock so concurrent client threads (FIFO writer, read pool)
    trigger one refresh, not a stampede. A failed refresh falls back to the
    last-known token (the facade then 401s and the client's never-raises
    contract soft-fails the operation) rather than raising into the hot path.
    """

    def __init__(self, authorizer):
        self._authorizer = authorizer
        self._refresh_lock = threading.Lock()
        # A fork can land while another thread holds the refresh lock; children
        # re-initialize it (same pattern as MetadataStoreClient's locks).
        mp_util.register_after_fork(self, MetadataStoreTokenProvider._after_fork_reinit)

    @staticmethod
    def _after_fork_reinit(provider: "MetadataStoreTokenProvider") -> None:
        provider._refresh_lock = threading.Lock()

    def __call__(self) -> str | None:
        token = self._authorizer.local_access_token
        if token is None or self._authorizer.does_token_need_refreshing(token):
            with self._refresh_lock:
                token = self._authorizer.local_access_token
                if token is None or self._authorizer.does_token_need_refreshing(token):
                    try:
                        self._authorizer.refresh_token()
                        token = self._authorizer.local_access_token
                    except Exception as exc:
                        logger.warning(
                            f"Metadata Store token refresh failed ({type(exc).__name__}: {exc}); "
                            f"using the last-known token"
                        )
        if token is None:
            return None
        return access_token_to_bearer(token)
