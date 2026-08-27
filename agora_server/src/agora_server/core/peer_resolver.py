# Copyright 2026 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
MetadataStorePeerResolver: resolves peer_id -> transport multiaddrs from the
Metadata Store address book, for injection into ``P2P`` (DHT->Redis migration M4b).

Once the libp2p Kademlia DHT is disabled (``dht_mode="none"``), the p2pd daemon
can no longer resolve a peer_id to its multiaddrs on its own. ``P2P`` calls this
resolver before dialing an unknown/expired peer and primes the daemon peerstore
with the returned addresses (``connect``).

The Metadata Store client is synchronous (httpx), so the GET runs in a thread
executor to avoid blocking the P2P event loop. The client is fork-aware (rebuilds
its connection pool when the pid changes), so a resolver built in a parent process
keeps working in forked consumer processes.
"""

from __future__ import annotations

import asyncio

from agora_server.core import address_book, envelope
from agora_server.core.metadata_store_client import MetadataStoreClient
from agora_server.hivemind.p2p import PeerID
from agora_server.hivemind.utils import get_dht_time
from agora_server.hivemind.utils.logging import get_logger
from agora_server.hivemind.utils.multiaddr import Multiaddr


logger = get_logger(__name__)


class MetadataStorePeerResolver:
    """Async ``peer_id -> (transport_maddrs, ttl_seconds)`` resolver over the Metadata Store.

    Returns ``None`` when the peer has no (live) address-book record — the caller
    then leaves the daemon peerstore untouched and lets the dial proceed/fail, so a
    not-yet-registered peer is simply retried on the next dial (no negative caching).

    Tests may inject ``http_client`` (a Starlette ``TestClient`` bound to an in-process
    Metadata Store) instead of opening a real connection.
    """

    def __init__(
        self,
        metadata_store_url: str | None = None,
        *,
        http_client=None,
        token: str | None = None,
        token_provider=None,
        require_signed_reads: bool = False,
    ):
        self._client = MetadataStoreClient(
            metadata_store_url, http_client=http_client, token=token, token_provider=token_provider
        )
        self._require_signed_reads = require_signed_reads

    async def __call__(self, peer_id: PeerID) -> tuple[list[Multiaddr], float] | None:
        # The Metadata Store client is blocking; run it off the P2P event loop.
        return await asyncio.get_running_loop().run_in_executor(None, self._fetch, peer_id)

    def _fetch(self, peer_id: PeerID) -> tuple[list[Multiaddr], float] | None:
        # Any failure here (store error, malformed envelope, bad multiaddr, wrong value
        # type) is treated as "unresolved" (return None) rather than crashing the dial
        # path -- the caller leaves the peerstore untouched and the next dial re-resolves.
        try:
            key = address_book.peer_address_book_key(peer_id)
            resp = self._client.get(key)
            if resp is None or resp.get("kind") != "single":
                return None  # absent, or (unexpectedly) a Pattern-B record
            try:
                # Verify the envelope signature AND that the signing key derives
                # the peer_id this record claims to describe (the key is
                # peer:{peer_id}). A dial primed from a forged record would fail
                # libp2p's identity check anyway; dropping it here avoids the
                # wasted dial and logs the forgery.
                maddr_strings, expiration = envelope.read(
                    resp["envelope"],
                    key=key,
                    expected_owner_peer_id=peer_id,
                    require_signed=self._require_signed_reads,
                )
            except envelope.EnvelopeVerificationError as e:
                logger.warning(f"Dropping address-book record for {peer_id}: {e}")
                return None
            if not isinstance(maddr_strings, list):
                logger.warning(f"address-book record for {peer_id} has unexpected value type {type(maddr_strings)}")
                return None
            maddrs = address_book.transport_maddrs(maddr_strings, peer_id)
            if not maddrs:
                return None
            ttl = max(0.0, expiration - get_dht_time())
            return maddrs, ttl
        except Exception as e:
            logger.debug(f"address-book lookup failed for {peer_id}: {e!r}")
            return None

    def close(self) -> None:
        self._client.close()
