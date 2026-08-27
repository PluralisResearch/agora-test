# This file contains code originally from Hivemind under MIT License
# Original: Copyright 2020 Learning@home authors and collaborators
# Modified by: Pluralis Research 2026
#
# Original code: MIT License (see THIRD_PARTY_LICENSES)
# Modifications: Apache 2.0 License (see LICENSE)
#
# Licensed under the Apache License, Version 2.0 (the "License") for modifications only;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""
RedisDHT: coordination metadata adapter backed by the Metadata Store API.

Wraps a real hivemind.DHT instance to retain the libp2p P2P transport layer
(needed for ConnectionHandler.replicate_p2p() and expert-to-expert tensor
transfer), while routing all KV coordination operations (store / get /
run_coroutine) over HTTP to the Metadata Store API instead of the Kademlia DHT.

Each value is wrapped in an envelope -- msgpack({payload, expiration, pubkey,
signature}), signed with the peer's persisted RSA identity key when one is
threaded in (unsigned envelopes otherwise) -- and the
Metadata Store stores it verbatim and applies the Redis Pattern A/B schema, the
run:{run_id}: prefix, and the validation pipeline. Reads verify the signature
and owner-tag binding and drop records that fail (see core/envelope.py). The
Pattern A/B storage logic that lived here in the PoC now lives server-side;
this module is purely a client: it builds/reads envelopes and maps the wire
responses back to the ValueWithExpiration shapes callers expect.
"""

import asyncio
import threading

from collections.abc import Callable, Sequence
from typing import Any

from agora_server.core import envelope
from agora_server.core.metadata_store_client import MetadataStoreClient
from agora_server.hivemind.dht import DHT
from agora_server.hivemind.utils import ValueWithExpiration, get_dht_time
from agora_server.hivemind.utils.crypto import RSAPrivateKey
from agora_server.hivemind.utils.logging import get_logger
from agora_server.hivemind.utils.mpfuture import MPFuture


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared store/get logic (used by both RedisDHT and RedisNode)
# ---------------------------------------------------------------------------


def _get(client: MetadataStoreClient, key: str, require_signed: bool = False) -> ValueWithExpiration | None:
    """Read a value, auto-detecting Pattern A (single) vs Pattern B (dict).

    Envelope signatures (and owner-tag bindings) are verified here; a record
    that fails verification is dropped — the whole read for Pattern A, the
    single entry for Pattern B — mirroring DHT-mode RSASignatureValidator
    semantics. Unsigned records are accepted, unless ``require_signed`` is set.
    """
    resp = client.get(key)
    if resp is None:
        return None
    if resp["kind"] == "single":
        try:
            value, expiration = envelope.read(resp["envelope"], key=key, require_signed=require_signed)
        except envelope.EnvelopeVerificationError as exc:
            logger.warning(f"Dropping Metadata Store record for {key!r}: {exc}")
            return None
        return ValueWithExpiration(value=value, expiration_time=expiration)

    # Pattern B: assemble a dict of subkey -> ValueWithExpiration.
    result: dict = {}
    max_expiry = get_dht_time()
    for entry in resp["entries"]:
        try:
            value, expiration = envelope.read(
                entry["envelope"], key=key, subkey=entry["subkey"], require_signed=require_signed
            )
        except envelope.EnvelopeVerificationError as exc:
            logger.warning(f"Dropping Metadata Store record for {key!r} / {entry['subkey']!r}: {exc}")
            continue
        result[entry["subkey"]] = ValueWithExpiration(value=value, expiration_time=expiration)
        if expiration > max_expiry:
            max_expiry = expiration
    if not result:
        # Every entry failed verification (or the index outlived all its lazily
        # reaped leaves): report the key as absent — a DHT would return None for
        # a record with no live content, and callers treat {} and None alike.
        return None
    return ValueWithExpiration(value=result, expiration_time=max_expiry)


def _get_many(
    client: MetadataStoreClient, keys: Sequence[str], require_signed: bool = False
) -> dict[str, ValueWithExpiration | None]:
    """Bulk read supporting BOTH Pattern A (single value) and Pattern B (subkey dict).

    Expert leaf records are owner-tagged (subkey = ``[owner:<pubkey>]``) and therefore stored as
    Pattern B, so a Pattern-A-only ``:mget`` would miss them (return None) and break expert lookups in
    Metadata Store mode. We fan out to per-key ``_get`` (which auto-detects Pattern A/B); each is one
    HTTP round-trip, but expert-uid lookups are small sets. ``found[uid].value`` is then a dict
    ``{subkey -> ValueWithExpiration}`` for owner-tagged leaves, matching the real DHT's get_many
    shape (see ``dht_handler.extract_leaf_record``).
    """
    if not keys:
        return {}
    return {key: _get(client, key, require_signed=require_signed) for key in keys}


# ---------------------------------------------------------------------------
# RedisNode — mimics DHTNode inside run_coroutine callbacks
# ---------------------------------------------------------------------------


class RedisNode:
    """
    Drop-in replacement for DHTNode passed to run_coroutine callbacks.

    Exposes async store / store_many / get / get_many methods backed by the
    Metadata Store. Unlike a real DHTNode it has no `.p2p`: the only callback that
    read libp2p (the address-book co-write in `_declare_experts`) detects the missing
    `.p2p` and instead sources the node's own maddrs from the cached
    `dht.get_visible_maddrs()` (SwarmP2P), so run_coroutine no longer attaches a P2P replica.

    The HTTP calls are synchronous (the shared MetadataStoreClient); they run on
    the dedicated run_coroutine thread, so blocking there is acceptable.

    Failure contract: like hivemind's DHTNode per-peer RPCs,
    these methods never raise on store/network failures — `get`/`get_many`
    return None entries, `store`/`store_many` return False entries. Failures
    are logged (and severity-classified) by the client.
    """

    num_workers: int | None = None

    def __init__(
        self,
        client: MetadataStoreClient,
        private_key: RSAPrivateKey | None = None,
        require_signed_reads: bool = False,
    ):
        self._client = client
        self._private_key = private_key
        self._require_signed_reads = require_signed_reads

    async def store(self, key: str, value: Any, expiration_time: float, subkey=None) -> bool:
        # Programming errors (unserializable values) raise regardless.
        env = envelope.build(value, expiration_time, key=key, subkey=subkey, private_key=self._private_key)
        try:
            return self._client.put(key, env, subkey=subkey)
        except Exception:
            return False  # already logged by the client

    async def store_many(
        self,
        keys: Sequence[str],
        values: Sequence[Any],
        expiration_time: float,
        subkeys: Sequence | None = None,
        num_workers: int | None = None,
    ) -> dict:
        """Bulk-store (key, subkey) -> value entries via the Metadata Store.

        Return-shape contract matches hivemind's DHTNode.store_many: the result
        is a dict keyed by ``(key, subkey)`` tuples when ``subkey is not None``,
        else just ``key``. Callers (e.g. ``_declare_experts``) rely on this shape.
        """
        if subkeys is None:
            subkeys = [None] * len(keys)
        # Envelopes built OUTSIDE the soft-fail: serialization errors are bugs.
        writes = []
        for key, value, subkey in zip(keys, values, subkeys, strict=True):
            env = envelope.build(value, expiration_time, key=key, subkey=subkey, private_key=self._private_key)
            write = {"key": key, "envelope": env}
            if subkey is not None:
                write["subkey"] = subkey
            writes.append(write)
        try:
            self._client.mput(writes)
            ok = True
        except Exception:
            ok = False  # already logged by the client
        return {(key, subkey) if subkey is not None else key: ok for key, subkey in zip(keys, subkeys, strict=True)}

    async def get(self, key: str, latest: bool = False) -> ValueWithExpiration | None:
        try:
            return _get(self._client, key, require_signed=self._require_signed_reads)
        except Exception:
            return None  # already logged by the client

    async def get_many(
        self,
        keys: Sequence[str],
        expiration_time: float | None = None,
        num_workers: int | None = None,
    ) -> dict[str, ValueWithExpiration | None]:
        try:
            return _get_many(self._client, keys, require_signed=self._require_signed_reads)
        except Exception:
            return {str(key): None for key in keys}


# ---------------------------------------------------------------------------
# RedisDHT — public API
# ---------------------------------------------------------------------------


class RedisDHT:
    """
    hivemind.DHT replacement that stores coordination metadata via the Metadata
    Store API.

    All store / get / run_coroutine calls go over HTTP to the Metadata Store. An
    optional P2P transport (SwarmP2P, or a real hivemind.DHT) supplies libp2p
    connectivity for callers that need it (ConnectionHandler.replicate_p2p(), the
    expert-RPC dial path, get_visible_maddrs). Read-only consumers that only need K/V
    -- e.g. the health monitor -- pass p2p=None and run no libp2p at all; the
    libp2p passthroughs then raise.

    Usage::

        p2p = SwarmP2P(start=True, ...)  # or a real DHT(...)
        dht = RedisDHT(metadata_store_url="https://metadata.swarm.internal", p2p=p2p)
        # read-only (no libp2p transport):
        dht = RedisDHT(metadata_store_url=..., p2p=None)

    Tests may inject ``http_client`` (e.g. a Starlette TestClient bound to the
    in-process Metadata Store app) instead of opening a real connection.
    """

    def __init__(
        self,
        metadata_store_url: str | None = None,
        *,
        p2p: DHT | None = None,
        http_client=None,
        token: str | None = None,
        token_provider=None,
        private_key: RSAPrivateKey | None = None,
        require_signed_reads: bool = False,
        compress_requests: bool = False,
    ):
        self._client = MetadataStoreClient(
            metadata_store_url,
            http_client=http_client,
            token=token,
            token_provider=token_provider,
            compress_requests=compress_requests,
        )
        self._p2p = p2p
        self._private_key = private_key
        self._require_signed_reads = require_signed_reads
        # The [owner:<pubkey>] marker of the signing key. Owner-tag producers
        # (progress tracker, dht_handler) source their tag from here so it can
        # never diverge from the envelope pubkey.
        self._local_public_key = (
            envelope.owner_tag(private_key.get_public_key().to_bytes()) if private_key is not None else None
        )
        if private_key is None:
            logger.warning(
                "RedisDHT created without a signing key; writes will be unsigned "
                "(rejected once the Metadata Store enforces signing)"
            )
        if metadata_store_url is not None:
            logger.info(f"RedisDHT routing KV to Metadata Store at {metadata_store_url}")

    @property
    def local_public_key(self) -> bytes | None:
        """The b"[owner:<pubkey>]" marker of this client's signing key (None when
        unsigned) — same format as RSASignatureValidator.local_public_key."""
        return self._local_public_key

    # ------------------------------------------------------------------
    # P2P passthrough — delegate to the P2P transport for libp2p connectivity
    # ------------------------------------------------------------------

    def _require_p2p(self):
        if self._p2p is None:
            raise RuntimeError(
                "RedisDHT was created with p2p=None (read-only Metadata Store client); "
                "this libp2p operation needs a P2P transport (SwarmP2P or a real DHT)."
            )
        return self._p2p

    @property
    def peer_id(self):
        return self._require_p2p().peer_id

    @property
    def num_workers(self) -> None:
        return None

    @property
    def client_mode(self) -> bool:
        return self._require_p2p().client_mode

    def get_visible_maddrs(self, latest: bool = False):
        return self._require_p2p().get_visible_maddrs(latest=latest)

    async def replicate_p2p(self, fresh: bool = False):
        # The transport may be a plain hivemind DHT, whose replicate_p2p takes no
        # arguments; private replicas (fresh=True) exist only on SwarmP2P.
        if fresh:
            return await self._require_p2p().replicate_p2p(fresh=True)
        return await self._require_p2p().replicate_p2p()

    def is_alive(self) -> bool:
        return self._p2p.is_alive() if self._p2p is not None else True

    def run_in_background(self, await_ready: bool = True) -> None:
        pass  # P2P transport already started in Server.create

    def shutdown(self) -> None:
        try:
            if self._p2p is not None:
                self._p2p.shutdown()
        finally:
            # Always release the HTTP client even if the P2P transport shutdown raised.
            self._client.close()

    # ------------------------------------------------------------------
    # KV operations — all go to the Metadata Store
    # ------------------------------------------------------------------

    def add_validators(self, validators) -> None:
        """No-op: record integrity is enforced by the Metadata Store."""

    def store(
        self,
        key: str,
        value: Any,
        expiration_time: float,
        subkey=None,
        return_future: bool = False,
        raise_on_error: bool = False,
    ):
        """Write a value via the Metadata Store (Pattern A or B depending on subkey).

        ``return_future=True`` enqueues the write onto the
        client's FIFO writer thread and returns an ``MPFuture`` immediately -
        the hivemind ``DHT.store`` contract: the caller is never blocked by the
        HTTP round-trip. The future resolves ``True``/``False``,
        never an exception - failures are logged by the client.

        ``return_future=False`` stays a synchronous call.
        It returns ``False`` instead of raising on store/network
        failures unless ``raise_on_error=True`` (strict callers, e.g. health
        checks). Serialization errors (unsupported value types) always raise -
        those are bugs.
        """
        # Programming errors (unserializable values) raise regardless.
        env = envelope.build(value, expiration_time, key=key, subkey=subkey, private_key=self._private_key)
        if return_future:
            future: MPFuture = MPFuture()

            def _resolve(ok: bool) -> None:
                try:
                    future.set_result(ok)
                except Exception:
                    pass  # future was cancelled or already resolved

            try:
                self._client.put_async(key, env, subkey=subkey, expiration_time=expiration_time, on_done=_resolve)
            except Exception as exc:  # enqueue-time failure (e.g. client closed)
                if raise_on_error:
                    raise
                logger.warning(f"Metadata Store async store enqueue failed for {key!r}: {type(exc).__name__}: {exc}")
                _resolve(False)
            return future
        try:
            return self._client.put(key, env, subkey=subkey)
        except Exception:
            if raise_on_error:
                raise
            return False  # already logged (and severity-classified) by the client

    def get(
        self,
        key: str,
        latest: bool = False,
        return_future: bool = False,
        raise_on_error: bool = False,
    ):
        """Read a value via the Metadata Store (Pattern A or B auto-detected).

        Never raises on read/network failures - returns None (hivemind's DHT
        contract) unless ``raise_on_error=True`` (strict callers that must
        distinguish "absent" from "unreachable", e.g. the health monitor).

        ``return_future=True`` runs the read on the
        client's thread pool and returns an ``MPFuture`` immediately - the
        hivemind contract for event-loop callers (matchmaking's poll, the
        progress fetcher): awaiting yields the loop instead of blocking it.
        With ``raise_on_error=True`` a failure is set as the future's
        exception (surfacing at ``await``/``.result()``); otherwise the future
        resolves None.
        """
        if return_future:
            future: MPFuture = MPFuture()

            def _read() -> None:
                try:
                    result = _get(self._client, key, require_signed=self._require_signed_reads)
                except Exception as exc:
                    if raise_on_error:
                        try:
                            future.set_exception(exc)
                        except Exception:
                            pass  # future was cancelled or already resolved
                        return
                    result = None  # already logged (and severity-classified) by the client
                try:
                    future.set_result(result)
                except Exception:
                    pass  # future was cancelled or already resolved

            try:
                self._client.submit_read(_read)
            except Exception as exc:  # submit-time failure (e.g. client closed)
                if raise_on_error:
                    raise
                logger.warning(f"Metadata Store async get submit failed for {key!r}: {type(exc).__name__}: {exc}")
                try:
                    future.set_result(None)
                except Exception:
                    pass
            return future
        try:
            return _get(self._client, key, require_signed=self._require_signed_reads)
        except Exception:
            if raise_on_error:
                raise
            return None  # already logged (and severity-classified) by the client

    def run_coroutine(self, coro: Callable, return_future: bool = False):
        """Execute an async coroutine that accepts (dht, node) positional args.

        All run_coroutine callers in the codebase are regular threading.Thread
        background threads without their own event loop, so asyncio.run() is safe.

        We construct a fresh RedisNode per call (which also avoids races between
        concurrent run_coroutine threads). Callbacks use only the node's K/V methods
        (store/store_many/get/get_many); none need libp2p, so no P2P replica is
        attached. (Expert-RPC transport still uses `RedisDHT.replicate_p2p()`
        directly, outside run_coroutine.)
        """

        async def _run():
            node = RedisNode(self._client, self._private_key, require_signed_reads=self._require_signed_reads)
            return await coro(self, node)

        if return_future:
            future: MPFuture = MPFuture()

            def _in_thread():
                try:
                    future.set_result(asyncio.run(_run()))
                except Exception as exc:
                    if not future.cancelled():
                        future.set_exception(exc)

            threading.Thread(target=_in_thread, daemon=True).start()
            return future
        else:
            return asyncio.run(_run())
