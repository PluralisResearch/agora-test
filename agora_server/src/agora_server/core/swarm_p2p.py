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
SwarmP2P: a standalone libp2p transport holder with NO Kademlia DHT.

Background. Today the libp2p ``P2P`` daemon is started and held by the
``hivemind.DHT`` process; every transport consumer (moe ``ConnectionHandler``,
``DecentralizedAverager``, ``RemoteExpertWorker``) obtains its ``P2P`` via
``dht.replicate_p2p()``, which replicates the one daemon that the DHT process
spawned. The Kademlia DHT layer running on top of that daemon is what resolves
``peer_id -> multiaddr`` (FIND_PEER).

In the DHT->Redis migration (docs/dht-wip/dht-redis-impl-m3-m4.md) that
resolution moves to the Metadata Store address book, so the Kademlia layer is no
longer needed. ``SwarmP2P`` is the replacement holder: it starts the p2pd daemon
with ``dht_mode="none"`` (no Kademlia at all) and exposes exactly the surface
``RedisDHT`` uses from its ``p2p`` transport -- ``peer_id``, ``client_mode``,
``get_visible_maddrs``, ``daemon_listen_maddr``, ``replicate_p2p``, ``is_alive``,
``shutdown`` -- so it is a drop-in for the ``p2p`` transport at the transport layer,
minus all K/V.

It mirrors ``hivemind.DHT``'s process model: a background ``ForkProcess`` owns
the daemon for its lifetime; consumers in other processes replicate the daemon
over its unix socket. Peer discovery is supplied externally (the Metadata Store
address book + an explicit ``connect`` prime before each dial; the resolver hook
lands in M4).
"""

from __future__ import annotations

import asyncio
import inspect
import multiprocessing as mp
import os
import signal

from contextlib import suppress

# MUST be a top-level import, not deferred into _build_resolver():
# replicate_p2p() runs inside fork-spawned children (ConnectionHandler,
# averager). A function-level import there deadlocks if the fork snapshotted
# the parent while another thread held this module chain's import lock — the
# child inherits a held lock with no thread to release it and blocks forever
# in importlib (observed: volunteers hanging at "Server started with 1
# modules:", never starting runtime pools, stuck in ghost phase 1).
from agora_server.core.peer_resolver import MetadataStorePeerResolver
from agora_server.hivemind.p2p import P2P, PeerID
from agora_server.hivemind.utils import MPFuture, get_logger, switch_to_uvloop
from agora_server.hivemind.utils.multiaddr import Multiaddr


logger = get_logger(__name__)

# How long the holder's idle loop blocks waiting for a control message before
# looping (so a shutdown is acted on promptly). Mirrors the role DHTNode's
# wait_timeout plays in hivemind.DHT.run.
_IDLE_POLL_TIMEOUT = 1.0


class SwarmP2P(mp.context.ForkProcess):
    """Standalone libp2p transport holder (no Kademlia DHT).

    Usage::

        p2p_holder = SwarmP2P(start=True, identity_path=..., host_maddrs=[...], announce_maddrs=[...])
        dht = RedisDHT(metadata_store_url=..., p2p=p2p_holder)

    :param start: if True, start the background process on construction.
    :param daemon: mark the background process as daemon (terminated with parent).
    :param shutdown_timeout: seconds to wait on ``.shutdown`` before terminating.
    :param await_ready: if True, the constructor blocks until the daemon is up.
    :param client_mode: reported via the ``client_mode`` property for callers
        that branch on it (no Kademlia client/server distinction exists here;
        it is purely advisory and defaults to False).
    :param p2p_kwargs: forwarded to ``P2P.create`` (e.g. ``identity_path``,
        ``host_maddrs``, ``announce_maddrs``, ``use_relay``, ``announce_maddrs``).
        ``dht_mode`` is forced to ``"none"`` and ``initial_peers`` is disallowed
        (there is no DHT to bootstrap).
    """

    def __init__(
        self,
        *,
        start: bool,
        daemon: bool = True,
        shutdown_timeout: float = 3,
        await_ready: bool = True,
        client_mode: bool = False,
        startup_timeout: float = 30,
        metadata_store_url: str | None = None,
        metadata_store_token: str | None = None,
        metadata_store_token_provider=None,
        metadata_store_http_client=None,
        metadata_store_require_signed_reads: bool = False,
        **p2p_kwargs,
    ):
        self._parent_pid = os.getpid()
        self._origin_pid = os.getpid()
        super().__init__()

        if p2p_kwargs.get("initial_peers") or p2p_kwargs.get("use_ipfs"):
            # Both set need_bootstrap=True in P2P.create, which also re-enables the
            # DHT-dependent is_identity_taken() probe -- neither makes sense without a DHT.
            raise ValueError("SwarmP2P runs without a DHT, so initial_peers/use_ipfs (bootstrap) is not supported")
        # Callers forward a DHT-shaped kwargs blob (host_maddrs, announce_maddrs,
        # identity_path, use_relay, ... PLUS DHT-only keys like record_validators,
        # num_workers, authorizer). Keep only what P2P.create accepts; the rest are
        # irrelevant to a transport-only holder (no Kademlia). Log the dropped keys so
        # this is never a silent surprise.
        accepted = set(inspect.signature(P2P.create).parameters)
        dropped = sorted(k for k in p2p_kwargs if k not in accepted)
        for k in dropped:
            p2p_kwargs.pop(k)
        if dropped:
            logger.debug(f"SwarmP2P ignoring non-P2P kwargs (DHT-only): {dropped}")
        # Force no-Kademlia regardless of what the caller passed.
        p2p_kwargs["dht_mode"] = "none"
        # The daemon-boot bound belongs to P2P.create (it waits for p2pd to come up);
        # forward our startup_timeout there (P2P.create's own default is only 15s). The
        # parent-side _ready wait then gets a margin on top so it doesn't fire while the
        # child is still within its own boot window.
        p2p_kwargs.setdefault("startup_timeout", startup_timeout)
        self._p2p_kwargs = p2p_kwargs

        # Peer discovery for the no-DHT daemon: each replica gets a process-local resolver
        # over the Metadata Store address book, injected into its P2P (M4b). When no store is
        # configured the resolver is None and dials rely on whatever is in the peerstore.
        self._metadata_store_url = metadata_store_url
        self._metadata_store_token = metadata_store_token
        self._metadata_store_token_provider = metadata_store_token_provider
        self._metadata_store_http_client = metadata_store_http_client
        self._metadata_store_require_signed_reads = metadata_store_require_signed_reads

        self._client_mode = client_mode
        self.shutdown_timeout = shutdown_timeout
        # Parent-side wait for the child to finish booting p2pd AND report readiness;
        # give it a margin over the daemon-boot timeout forwarded to P2P.create above.
        self._startup_timeout = startup_timeout + 15
        self._inner_pipe, self._outer_pipe = mp.Pipe(duplex=False)
        self._ready: MPFuture = MPFuture()
        self.daemon = daemon

        # Populated from the child process via _ready (immutable for the daemon's lifetime).
        self._peer_id: PeerID | None = None
        self._daemon_listen_maddr: Multiaddr | None = None
        self._visible_maddrs: list[Multiaddr] | None = None

        # Per-process replica cache (rebuilt when the accessing pid changes, like hivemind.DHT).
        self._p2p_replica: P2P | None = None

        if start:
            self.run_in_background(await_ready=await_ready)

    # ------------------------------------------------------------------
    # Child process
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Own the p2pd daemon for the lifetime of this process (child side)."""
        loop = switch_to_uvloop()
        pipe_semaphore = asyncio.Semaphore(value=0)
        loop.add_reader(self._inner_pipe.fileno(), pipe_semaphore.release)

        async def _run():
            # Match hivemind.DHT: ignore SIGINT in the child; the parent drives shutdown.
            signal.signal(signal.SIGINT, signal.SIG_IGN)

            try:
                self._p2p = await P2P.create(**self._p2p_kwargs)
            except Exception as e:
                logger.debug(e, exc_info=True)
                self._ready.set_exception(e)
                return
            try:
                visible = await self._p2p.get_visible_maddrs()
            except Exception as e:
                # The daemon started but is unusable (e.g. no visible maddrs). Tear it
                # down explicitly rather than leaking the p2pd subprocess until exit.
                logger.debug(e, exc_info=True)
                with suppress(Exception):
                    await self._p2p.shutdown()
                self._ready.set_exception(e)
                return

            self._ready.set_result(
                {
                    "peer_id": self._p2p.peer_id,
                    "daemon_listen_maddr": str(self._p2p.daemon_listen_maddr),
                    "visible_maddrs": [str(maddr) for maddr in visible],
                }
            )

            while True:
                try:
                    await asyncio.wait_for(pipe_semaphore.acquire(), timeout=_IDLE_POLL_TIMEOUT)
                except asyncio.TimeoutError:
                    pass
                if not self._inner_pipe.poll():
                    continue
                try:
                    method, args, kwargs = self._inner_pipe.recv()
                except (OSError, ConnectionError, RuntimeError) as e:
                    logger.exception(e)
                    await asyncio.sleep(_IDLE_POLL_TIMEOUT)
                    continue
                task = asyncio.create_task(getattr(self, method)(*args, **kwargs))
                if method == "_shutdown":
                    await task
                    break

        loop.run_until_complete(_run())
        loop.close()

    async def _shutdown(self) -> None:
        await self._p2p.shutdown()

    # ------------------------------------------------------------------
    # Parent process: lifecycle
    # ------------------------------------------------------------------

    def run_in_background(self, await_ready: bool = True, timeout: float | None = None) -> None:
        """Start the holder process; if await_ready, block until the daemon is up."""
        self.start()
        if await_ready:
            self.wait_until_ready(timeout if timeout is not None else self._startup_timeout)

    def wait_until_ready(self, timeout: float | None = None) -> None:
        self._ensure_ready(timeout)

    def _ensure_ready(self, timeout: float | None = None) -> None:
        if self._peer_id is not None:
            return
        if os.getpid() != self._parent_pid:
            # MPFuture.result() can only be consumed by the origin process. With the
            # default await_ready=True the parent caches peer_id/maddrs before any
            # consumer forks (so this branch is never hit); guard the await_ready=False
            # anti-pattern with a clear error instead of an opaque MPFuture failure.
            raise RuntimeError(
                "SwarmP2P accessed from a non-origin process before it was ready; "
                "construct it with await_ready=True (the default) in the parent before forking consumers"
            )
        info = self._ready.result(timeout=timeout)
        self._peer_id = info["peer_id"]
        self._daemon_listen_maddr = Multiaddr(info["daemon_listen_maddr"])
        self._visible_maddrs = [Multiaddr(s) for s in info["visible_maddrs"]]

    def shutdown(self) -> None:
        """Shut down the holder process (and its p2pd daemon)."""
        if self.is_alive():
            self._outer_pipe.send(("_shutdown", [], {}))
            self._outer_pipe.close()
            self.join(self.shutdown_timeout)
            if self.is_alive():
                logger.warning("SwarmP2P did not shut down within the grace period; terminating it the hard way")
                self.terminate()

    def __del__(self):
        if self._parent_pid == os.getpid() and self.is_alive():
            self.shutdown()

    # ------------------------------------------------------------------
    # Parent process: transport surface (drop-in for the inner DHT)
    # ------------------------------------------------------------------

    @property
    def peer_id(self) -> PeerID:
        self._ensure_ready()
        return self._peer_id

    @property
    def client_mode(self) -> bool:
        return self._client_mode

    @property
    def daemon_listen_maddr(self) -> Multiaddr:
        self._ensure_ready()
        return self._daemon_listen_maddr

    def get_visible_maddrs(self, latest: bool = False) -> list[Multiaddr]:
        """Multiaddrs other peers can use to reach this node.

        The ``latest`` kwarg is accepted for signature-compatibility with
        ``hivemind.DHT.get_visible_maddrs`` but is a no-op: the daemon's visible
        multiaddrs are captured at startup and are stable for our static
        host/announce configuration. Callers that need a live refresh can use
        ``replicate_p2p()`` and call ``await p2p.get_visible_maddrs(latest=True)``.
        """
        self._ensure_ready()
        return list(self._visible_maddrs)

    async def replicate_p2p(self, fresh: bool = False) -> P2P:
        """Return a ``P2P`` replica bound to this node's daemon for the calling process.

        Mirrors ``hivemind.DHT.replicate_p2p``: one cached replica per ``origin_pid``
        so a forked child rebuilds its own (the parent's replica tasks do not
        survive a fork). Each replica gets a process-local Metadata Store peer
        resolver (when a store is configured) so the no-DHT daemon can resolve dials.

        The cached replica's IO is bound to the event loop of whichever caller created
        it first, so it is only safe to await from that loop: control-socket requests
        submitted from any other loop are never pumped and hang until the caller's
        timeout. A component that runs its own event loop must pass ``fresh=True`` to
        get a private, uncached replica bound to that loop (and owns closing it).
        """
        if fresh:
            return await P2P.replicate(self.daemon_listen_maddr, peer_resolver=self._build_resolver())
        if self._p2p_replica is None or self._origin_pid != os.getpid():
            self._origin_pid = os.getpid()
            self._p2p_replica = await P2P.replicate(self.daemon_listen_maddr, peer_resolver=self._build_resolver())
        return self._p2p_replica

    def _build_resolver(self):
        if self._metadata_store_url is None and self._metadata_store_http_client is None:
            return None
        return MetadataStorePeerResolver(
            self._metadata_store_url,
            http_client=self._metadata_store_http_client,
            token=self._metadata_store_token,
            token_provider=self._metadata_store_token_provider,
            require_signed_reads=self._metadata_store_require_signed_reads,
        )
