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

import asyncio
import json
import logging
import os
import secrets
import time
import warnings
import weakref

from collections import OrderedDict
from collections.abc import AsyncIterable as AsyncIterableABC
from collections.abc import Awaitable, Callable, Sequence
from contextlib import closing, suppress
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import path
from typing import Any, AsyncIterator, List, Optional, Tuple, Type, TypeVar, Union

from google.protobuf.message import Message

import agora_server.hivemind._bin as cli  # vendored home of the p2pd daemon binary
import agora_server.hivemind.p2p.p2p_daemon_bindings.p2pclient as p2pclient

from agora_server.hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, P2PDaemonError, P2PHandlerError
from agora_server.hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID, PeerInfo, StreamInfo
from agora_server.hivemind.p2p.p2p_daemon_bindings.utils import ControlFailure
from agora_server.hivemind.proto import crypto_pb2
from agora_server.hivemind.proto.p2pd_pb2 import RPCError
from agora_server.hivemind.utils.asyncio import as_aiter, asingle, cancel_task_if_running
from agora_server.hivemind.utils.crypto import RSAPrivateKey
from agora_server.hivemind.utils.logging import get_logger, golog_level_to_python, loglevel, python_level_to_golog
from agora_server.hivemind.utils.multiaddr import Multiaddr


logger = get_logger(__name__)


P2PD_FILENAME = "p2pd"


@dataclass(frozen=True)
class P2PContext:
    handle_name: str
    local_id: PeerID
    remote_id: PeerID = None


class P2P:
    """
    This class is responsible for establishing peer-to-peer connections through NAT and/or firewalls.
    It creates and manages a libp2p daemon (https://libp2p.io) in a background process,
    then terminates it when P2P is shut down. In order to communicate, a P2P instance should
    either use one or more initial_peers that will connect it to the rest of the swarm or
    use the public IPFS network (https://ipfs.io).

    For incoming connections, P2P instances add RPC handlers that may be accessed by other peers:
      - `P2P.add_protobuf_handler` accepts a protobuf message and returns another protobuf
      - `P2P.add_binary_stream_handler` transfers raw data using bi-directional streaming interface

    To access these handlers, a P2P instance can `P2P.call_protobuf_handler`/`P2P.call_binary_stream_handler`,
    using the recipient's unique `P2P.peer_id` and the name of the corresponding handler.
    """

    HEADER_LEN = 8
    BYTEORDER = "big"
    MESSAGE_MARKER = b"\x00"
    ERROR_MARKER = b"\x01"
    END_OF_STREAM = RPCError()

    DHT_MODE_MAPPING = {
        "auto": {"dht": 1},
        "server": {"dhtServer": 1},
        "client": {"dhtClient": 1},
        # "none" launches p2pd with no -dht/-dhtClient/-dhtServer flag at all, i.e.
        # no libp2p Kademlia DHT. Peer discovery must then be supplied externally
        # (the Metadata Store address book + explicit connect priming); see the
        # DHT->Redis migration M3/M4 (docs/dht-wip/dht-redis-impl-m3-m4.md).
        "none": {},
    }
    FORCE_REACHABILITY_MAPPING = {
        "public": {"forceReachabilityPublic": 1},
        "private": {"forceReachabilityPrivate": 1},
    }
    _UNIX_SOCKET_PREFIX = "/unix/tmp/hivemind-"
    # Safety bound on the peer-resolution prime cache (M4b). A node dials at most a few
    # dozen peers; this cap only matters under pathological long-run churn. Eviction is
    # harmless — it just triggers a re-resolve on the next dial.
    _PRIMED_CACHE_MAX = 4096

    def __init__(self):
        self.peer_id = None
        self._client = None
        self._child = None
        self._alive = False
        self._reader_task = None
        self._listen_task = None
        # Peer-id -> multiaddr resolution hook (DHT->Redis migration M4b). When set, the
        # daemon runs without a Kademlia DHT and we prime its peerstore via connect()
        # before dialing. None => resolution is the daemon's own DHT (legacy behavior),
        # so the hook is a no-op and existing dial paths are unchanged.
        self._peer_resolver: Callable[[PeerID], Awaitable] | None = None
        self._dial_timeout: float = 10.0
        # peer_id -> monotonic deadline until which the prime is fresh. LRU-capped so the
        # cache can't grow unbounded over a long run with churn; evicting an entry is
        # harmless (it just triggers a re-resolve on the next dial).
        self._primed: OrderedDict[PeerID, float] = OrderedDict()
        # peer_id -> asyncio.Lock (coalesces concurrent re-primes of the same peer). Weak
        # values so a lock lives only while a coroutine is holding/awaiting it (that holder
        # keeps a strong ref across `async with`); idle locks are GC'd -> bounded to the
        # concurrent in-flight count.
        self._primed_locks: weakref.WeakValueDictionary[PeerID, asyncio.Lock] = weakref.WeakValueDictionary()

    @classmethod
    async def create(
        cls,
        initial_peers: Sequence[Multiaddr | str] | None = None,
        *,
        announce_maddrs: Sequence[Multiaddr | str] | None = None,
        auto_nat: bool = True,
        conn_manager: bool = True,
        dht_mode: str = "server",
        force_reachability: str | None = None,
        host_maddrs: Sequence[Multiaddr | str] | None = ("/ip4/127.0.0.1/tcp/0",),
        identity_path: str | None = None,
        idle_timeout: float = 30,
        nat_port_map: bool = True,
        relay_hop_limit: int = 0,
        startup_timeout: float = 15,
        tls: bool = True,
        use_auto_relay: bool = False,
        use_ipfs: bool = False,
        use_relay: bool = True,
        persistent_conn_max_msg_size: int = DEFAULT_MAX_MSG_SIZE,
        quic: bool | None = None,
        use_relay_hop: bool | None = None,
        use_relay_discovery: bool | None = None,
        check_if_identity_free: bool = True,
        no_listen: bool = False,
        trusted_relays: Sequence[Multiaddr | str] | None = None,
        peer_resolver: Callable[["PeerID"], Awaitable] | None = None,
        dial_timeout: float = 10.0,
    ) -> "P2P":
        """
        Start a new p2pd process and connect to it.
        :param initial_peers: List of bootstrap peers
        :param auto_nat: Enables the AutoNAT service
        :param announce_maddrs: Visible multiaddrs that the peer will announce
                                for external connections from other p2p instances
        :param conn_manager: Enables the Connection Manager
        :param dht_mode: libp2p DHT mode (auto/client/server/none).
                         Defaults to "server" to make collaborations work in local networks.
                         "none" disables the libp2p Kademlia DHT entirely (no -dht flag); peer
                         discovery must then be supplied externally (Metadata Store address book).
                         Details: https://pkg.go.dev/github.com/libp2p/go-libp2p-kad-dht#ModeOpt
        :param force_reachability: Force reachability mode (public/private)
        :param host_maddrs: Multiaddrs to listen for external connections from other p2p instances
        :param identity_path: Path to a private key file. If defined, makes the peer ID deterministic.
                              If the file does not exist yet, writes a new private key to this file.
        :param idle_timeout: kill daemon if client has been idle for a given number of
                             seconds before opening persistent streams
        :param nat_port_map: Enables NAT port mapping
        :param relay_hop_limit: sets the hop limit for hop relays
        :param startup_timeout: raise a P2PDaemonError if the daemon does not start in ``startup_timeout`` seconds
        :param tls: Enables TLS1.3 channel security protocol
        :param use_ipfs: Bootstrap to IPFS (incompatible with initial_peers)
        :param use_relay: Enable circuit relay functionality in libp2p
                          (see https://docs.libp2p.io/concepts/nat/circuit-relay/).
                          If enabled (default), you can reach peers behind NATs/firewalls through libp2p relays.
                          If you are behind NAT/firewall yourself,
                          please pass `use_auto_relay=True` to become reachable.
        :param use_auto_relay: Look for libp2p relays to become reachable if we are behind NAT/firewall
        :param quic: Deprecated, has no effect since libp2p 0.17.0
        :param use_relay_hop: Deprecated, has no effect since libp2p 0.17.0
        :param use_relay_discovery: Deprecated, has no effect since libp2p 0.17.0
        :param check_if_identity_free: If enabled (default), ``identity_path`` is provided,
                                       and we are connecting to an existing swarm,
                                       ensure that this identity is not used by other peers already.
                                       This slows down ``P2P.create()`` but protects from unintuitive libp2p errors
                                       appearing in case of the identity collision.
        :return: a wrapper for the p2p daemon
        """

        assert not (initial_peers and use_ipfs), (
            "User-defined initial_peers and use_ipfs=True are incompatible, please choose one option"
        )

        if not all(arg is None for arg in [quic, use_relay_hop, use_relay_discovery]):
            warnings.warn(
                "Parameters `quic`, `use_relay_hop`, and `use_relay_discovery` of hivemind.P2P "
                "have no effect since libp2p 0.17.0 and will be removed in hivemind 1.2.0+",
                DeprecationWarning,
                stacklevel=2,
            )

        self = cls()
        self._peer_resolver = peer_resolver
        self._dial_timeout = dial_timeout
        with path(cli, P2PD_FILENAME) as p:
            p2pd_path = p

        socket_uid = secrets.token_urlsafe(8)
        self._daemon_listen_maddr = Multiaddr(cls._UNIX_SOCKET_PREFIX + f"p2pd-{socket_uid}.sock")
        self._client_listen_maddr = Multiaddr(cls._UNIX_SOCKET_PREFIX + f"p2pclient-{socket_uid}.sock")
        if announce_maddrs is not None:
            for addr in announce_maddrs:
                addr = Multiaddr(addr)
                if ("tcp" in addr and addr["tcp"] == "0") or ("udp" in addr and addr["udp"] == "0"):
                    raise ValueError("Please specify an explicit port in announce_maddrs: port 0 is not supported")

        need_bootstrap = bool(initial_peers) or use_ipfs
        process_kwargs = cls.DHT_MODE_MAPPING[dht_mode].copy()
        process_kwargs.update(cls.FORCE_REACHABILITY_MAPPING.get(force_reachability, {}))
        for param, value in [
            ("bootstrapPeers", initial_peers),
            ("hostAddrs", host_maddrs),
            ("announceAddrs", announce_maddrs),
            ("trustedRelays", trusted_relays),
        ]:
            if value:
                process_kwargs[param] = self._maddrs_to_str(value)
        if no_listen:
            process_kwargs["noListenAddrs"] = 1
        if identity_path is not None:
            if os.path.isfile(identity_path):
                if check_if_identity_free and need_bootstrap:
                    logger.info(f"Checking that identity from `{identity_path}` is not used by other peers")
                    if await cls.is_identity_taken(
                        identity_path,
                        initial_peers=initial_peers,
                        tls=tls,
                        use_auto_relay=use_auto_relay,
                        use_ipfs=use_ipfs,
                        use_relay=use_relay,
                    ):
                        raise P2PDaemonError(f"Identity from `{identity_path}` is already taken by another peer")
            else:
                logger.info(f"Generating new identity to be saved in `{identity_path}`")
                self.generate_identity(identity_path)
                # A newly generated identity is not taken with ~100% probability
            process_kwargs["id"] = identity_path

        proc_args = self._make_process_args(
            str(p2pd_path),
            autoRelay=use_auto_relay,
            autonat=auto_nat,
            b=need_bootstrap,
            connManager=conn_manager,
            idleTimeout=f"{idle_timeout}s",
            listen=self._daemon_listen_maddr,
            natPortMap=nat_port_map,
            relay=use_relay,
            relayHopLimit=relay_hop_limit,
            tls=tls,
            persistentConnMaxMsgSize=persistent_conn_max_msg_size,
            **process_kwargs,
        )

        env = os.environ.copy()
        env.setdefault("GOLOG_LOG_LEVEL", python_level_to_golog(loglevel))
        env["GOLOG_LOG_FMT"] = "json"

        logger.debug(f"Launching {proc_args}")
        self._child = await asyncio.subprocess.create_subprocess_exec(
            *proc_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
        )
        self._alive = True

        ready = asyncio.Future()
        self._reader_task = asyncio.create_task(self._read_outputs(ready))
        try:
            await asyncio.wait_for(ready, startup_timeout)
        except asyncio.TimeoutError:
            await self.shutdown()
            raise P2PDaemonError(f"Daemon failed to start in {startup_timeout:.1f} seconds")

        self._client = await p2pclient.Client.create(
            control_maddr=self._daemon_listen_maddr,
            listen_maddr=self._client_listen_maddr,
            persistent_conn_max_msg_size=persistent_conn_max_msg_size,
        )

        await self._ping_daemon()
        return self

    @classmethod
    async def is_identity_taken(
        cls,
        identity_path: str,
        *,
        initial_peers: Sequence[Multiaddr | str] | None,
        tls: bool,
        use_auto_relay: bool,
        use_ipfs: bool,
        use_relay: bool,
    ) -> bool:
        with open(identity_path, "rb") as f:
            peer_id = PeerID.from_identity(f.read())

        anonymous_p2p = await cls.create(
            initial_peers=initial_peers,
            dht_mode="client",
            tls=tls,
            use_auto_relay=use_auto_relay,
            use_ipfs=use_ipfs,
            use_relay=use_relay,
        )
        try:
            await anonymous_p2p._client.connect(peer_id, [])
            return True
        except ControlFailure:
            return False
        finally:
            await anonymous_p2p.shutdown()

    @staticmethod
    def generate_identity(identity_path: str) -> None:
        private_key = RSAPrivateKey()
        protobuf = crypto_pb2.PrivateKey(key_type=crypto_pb2.KeyType.RSA, data=private_key.to_bytes())

        try:
            with open(identity_path, "wb") as f:
                f.write(protobuf.SerializeToString())
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The directory `{os.path.dirname(identity_path)}` for saving the identity does not exist"
            )
        os.chmod(identity_path, 0o400)

    @classmethod
    async def replicate(
        cls,
        daemon_listen_maddr: Multiaddr,
        peer_resolver: Callable[["PeerID"], Awaitable] | None = None,
        dial_timeout: float = 10.0,
    ) -> "P2P":
        """
        Connect to existing p2p daemon
        :param daemon_listen_maddr: multiaddr of the existing p2p daemon
        :param peer_resolver: optional peer_id -> (maddrs, ttl) hook used to prime the
                              daemon peerstore before dialing when the daemon has no DHT (M4b)
        :return: new wrapper for the existing p2p daemon
        """

        self = cls()
        self._peer_resolver = peer_resolver
        self._dial_timeout = dial_timeout
        # There is no child under control
        # Use external already running p2pd
        self._child = None
        self._alive = True

        socket_uid = secrets.token_urlsafe(8)
        self._daemon_listen_maddr = daemon_listen_maddr
        self._client_listen_maddr = Multiaddr(cls._UNIX_SOCKET_PREFIX + f"p2pclient-{socket_uid}.sock")

        self._client = await p2pclient.Client.create(self._daemon_listen_maddr, self._client_listen_maddr)

        await self._ping_daemon()
        return self

    async def _ping_daemon(self) -> None:
        self.peer_id, self._visible_maddrs = await self._client.identify()
        logger.debug(f"Launched p2pd with peer id = {self.peer_id}, host multiaddrs = {self._visible_maddrs}")

    async def get_visible_maddrs(self, latest: bool = False) -> list[Multiaddr]:
        """
        Get multiaddrs of the current peer that should be accessible by other peers.

        :param latest: ask the P2P daemon to refresh the visible multiaddrs
        """

        if latest:
            _, self._visible_maddrs = await self._client.identify()

        if not self._visible_maddrs:
            raise ValueError(f"No multiaddrs found for peer {self.peer_id}")

        p2p_maddr = Multiaddr(f"/p2p/{self.peer_id.to_base58()}")
        return [addr.encapsulate(p2p_maddr) for addr in self._visible_maddrs]

    async def list_peers(self) -> list[PeerInfo]:
        return list(await self._client.list_peers())

    @property
    def daemon_listen_maddr(self) -> Multiaddr:
        return self._daemon_listen_maddr

    @staticmethod
    async def send_raw_data(data: bytes, writer: asyncio.StreamWriter, *, chunk_size: int = 2**16) -> None:
        writer.write(len(data).to_bytes(P2P.HEADER_LEN, P2P.BYTEORDER))
        data = memoryview(data)
        for offset in range(0, len(data), chunk_size):
            writer.write(data[offset : offset + chunk_size])
        await writer.drain()

    @staticmethod
    async def receive_raw_data(reader: asyncio.StreamReader) -> bytes:
        header = await reader.readexactly(P2P.HEADER_LEN)
        content_length = int.from_bytes(header, P2P.BYTEORDER)
        data = await reader.readexactly(content_length)
        return data

    TInputProtobuf = TypeVar("TInputProtobuf")
    TOutputProtobuf = TypeVar("TOutputProtobuf")

    @staticmethod
    async def send_protobuf(protobuf: TOutputProtobuf | RPCError, writer: asyncio.StreamWriter) -> None:
        if isinstance(protobuf, RPCError):
            writer.write(P2P.ERROR_MARKER)
        else:
            writer.write(P2P.MESSAGE_MARKER)
        await P2P.send_raw_data(protobuf.SerializeToString(), writer)

    @staticmethod
    async def receive_protobuf(
        input_protobuf_type: type[Message], reader: asyncio.StreamReader
    ) -> tuple[TInputProtobuf | None, RPCError | None]:
        msg_type = await reader.readexactly(1)
        if msg_type == P2P.MESSAGE_MARKER:
            protobuf = input_protobuf_type()
            protobuf.ParseFromString(await P2P.receive_raw_data(reader))
            return protobuf, None
        elif msg_type == P2P.ERROR_MARKER:
            protobuf = RPCError()
            protobuf.ParseFromString(await P2P.receive_raw_data(reader))
            return None, protobuf
        else:
            raise TypeError("Invalid Protobuf message type")

    TInputStream = AsyncIterator[TInputProtobuf]
    TOutputStream = AsyncIterator[TOutputProtobuf]

    async def _add_protobuf_stream_handler(
        self,
        name: str,
        handler: Callable[[TInputStream, P2PContext], TOutputStream],
        input_protobuf_type: type[Message],
        max_prefetch: int = 5,
        balanced: bool = False,
    ) -> None:
        """
        :param max_prefetch: Maximum number of items to prefetch from the request stream.
          ``max_prefetch <= 0`` means unlimited.

        :note:  Since the cancel messages are sent via the input stream,
          they will not be received while the prefetch buffer is full.
        """

        async def _handle_stream(
            stream_info: StreamInfo, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            context = P2PContext(
                handle_name=name,
                local_id=self.peer_id,
                remote_id=stream_info.peer_id,
            )
            requests = asyncio.Queue(max_prefetch)

            async def _read_stream() -> P2P.TInputStream:
                while True:
                    request = await requests.get()
                    if request is None:
                        break
                    yield request

            async def _process_stream() -> None:
                try:
                    async for response in handler(_read_stream(), context):
                        try:
                            await P2P.send_protobuf(response, writer)
                        except Exception:
                            # The connection is unexpectedly closed by the caller or broken.
                            # The loglevel is DEBUG since the actual error will be reported on the caller
                            logger.debug("Exception while sending response:", exc_info=True)
                            break
                except Exception as e:
                    logger.warning("Handler failed with the exception:", exc_info=True)
                    with suppress(Exception):
                        # Sometimes `e` is a connection error, so it is okay if we fail to report `e` to the caller
                        await P2P.send_protobuf(RPCError(message=str(e)), writer)

            with closing(writer):
                processing_task = asyncio.create_task(_process_stream())
                try:
                    while True:
                        receive_task = asyncio.create_task(P2P.receive_protobuf(input_protobuf_type, reader))
                        await asyncio.wait({processing_task, receive_task}, return_when=asyncio.FIRST_COMPLETED)

                        if processing_task.done():
                            receive_task.cancel()
                            return

                        if receive_task.done():
                            try:
                                request, _ = await receive_task
                            except asyncio.IncompleteReadError:  # Connection is closed (the client cancelled or died)
                                return
                            await requests.put(request)  # `request` is None for the end-of-stream message
                except Exception:
                    logger.warning("Exception while receiving requests:", exc_info=True)
                finally:
                    processing_task.cancel()

        await self.add_binary_stream_handler(name, _handle_stream, balanced=balanced)

    async def _iterate_protobuf_stream_handler(
        self, peer_id: PeerID, name: str, requests: TInputStream, output_protobuf_type: type[Message]
    ) -> TOutputStream:
        _, reader, writer = await self.call_binary_stream_handler(peer_id, name)

        async def _write_to_stream() -> None:
            async for request in requests:
                await P2P.send_protobuf(request, writer)
            await P2P.send_protobuf(P2P.END_OF_STREAM, writer)

        async def _read_from_stream() -> AsyncIterator[Message]:
            with closing(writer):
                try:
                    while True:
                        try:
                            response, err = await P2P.receive_protobuf(output_protobuf_type, reader)
                        except asyncio.IncompleteReadError:  # Connection is closed
                            break

                        if err is not None:
                            raise P2PHandlerError(f"Failed to call handler `{name}` at {peer_id}: {err.message}")
                        yield response

                    await writing_task
                finally:
                    writing_task.cancel()

        writing_task = asyncio.create_task(_write_to_stream())
        return _read_from_stream()

    async def add_protobuf_handler(
        self,
        name: str,
        handler: Callable[[TInputProtobuf | TInputStream, P2PContext], Awaitable[TOutputProtobuf] | TOutputStream],
        input_protobuf_type: type[Message],
        *,
        stream_input: bool = False,
        stream_output: bool = False,
        balanced: bool = False,
    ) -> None:
        """
        :param stream_input: If True, assume ``handler`` to take ``TInputStream``
                             (not just ``TInputProtobuf``) as input.
        :param stream_output: If True, assume ``handler`` to return ``TOutputStream``
                              (not ``Awaitable[TOutputProtobuf]``).
        :param balanced: If True, handler will be balanced on p2pd side between all handlers in python.
                         Default: False
        """

        if not stream_input and not stream_output:
            await self._add_protobuf_unary_handler(name, handler, input_protobuf_type, balanced=balanced)
            return

        async def _stream_handler(requests: P2P.TInputStream, context: P2PContext) -> P2P.TOutputStream:
            input = requests if stream_input else await asingle(requests)
            output = handler(input, context)

            if isinstance(output, AsyncIterableABC):
                async for item in output:
                    yield item
            else:
                yield await output

        await self._add_protobuf_stream_handler(name, _stream_handler, input_protobuf_type, balanced=balanced)

    async def remove_protobuf_handler(
        self,
        name: str,
        *,
        stream_input: bool = False,
        stream_output: bool = False,
    ) -> None:
        if not stream_input and not stream_output:
            await self._client.remove_unary_handler(name)
            return

        await self.remove_binary_stream_handler(name)

    async def _add_protobuf_unary_handler(
        self,
        handle_name: str,
        handler: Callable[[TInputProtobuf, P2PContext], Awaitable[TOutputProtobuf]],
        input_protobuf_type: type[Message],
        balanced: bool = False,
    ) -> None:
        """
        Register a request-response (unary) handler. Unary requests and responses
        are sent through persistent multiplexed connections to the daemon for the
        sake of reducing the number of open files.
        :param handle_name: name of the handler (protocol id)
        :param handler: function handling the unary requests
        :param input_protobuf_type: protobuf type of the request
        """

        async def _unary_handler(request: bytes, remote_id: PeerID) -> bytes:
            input_serialized = input_protobuf_type.FromString(request)
            context = P2PContext(
                handle_name=handle_name,
                local_id=self.peer_id,
                remote_id=remote_id,
            )

            response = await handler(input_serialized, context)
            return response.SerializeToString()

        await self._client.add_unary_handler(handle_name, _unary_handler, balanced=balanced)

    async def call_protobuf_handler(
        self,
        peer_id: PeerID,
        name: str,
        input: TInputProtobuf | TInputStream,
        output_protobuf_type: type[Message],
    ) -> Awaitable[TOutputProtobuf]:
        if not isinstance(input, AsyncIterableABC):
            return await self._call_unary_protobuf_handler(peer_id, name, input, output_protobuf_type)

        responses = await self._iterate_protobuf_stream_handler(peer_id, name, input, output_protobuf_type)
        return await asingle(responses)

    async def _ensure_peer_connected(self, peer_id: PeerID, *, force_refresh: bool = False) -> None:
        """Prime the daemon peerstore with ``peer_id``'s multiaddrs from the resolver (M4b).

        No-op when no resolver is configured: resolution then comes from the daemon's own
        DHT, exactly as before. With a resolver (no-DHT daemon), we ``connect`` explicitly
        before dialing. A successful prime is cached until the address-book record's TTL
        elapses; ``force_refresh`` bypasses the cache after a dial failure.
        """
        if self._peer_resolver is None:
            return
        if not force_refresh:
            deadline = self._primed.get(peer_id)
            if deadline is not None and time.monotonic() < deadline:
                return
        # Get-or-create the per-peer lock. WeakValueDictionary: holding `lock` keeps it
        # alive across the `async with`; once no coroutine references it, it is GC'd.
        lock = self._primed_locks.get(peer_id)
        if lock is None:
            lock = asyncio.Lock()
            self._primed_locks[peer_id] = lock
        async with lock:
            if not force_refresh:
                deadline = self._primed.get(peer_id)
                if deadline is not None and time.monotonic() < deadline:
                    return
            resolved = await self._peer_resolver(peer_id)
            if resolved is None:
                # Not in the address book (yet). Drop any stale prime and let the dial
                # proceed/fail; the next attempt re-resolves (no negative caching).
                self._primed.pop(peer_id, None)
                return
            maddrs, ttl = resolved
            try:
                await asyncio.wait_for(self._client.connect(peer_id, maddrs), timeout=self._dial_timeout)
            except Exception:
                # connect failed (incl. timeout): drop any cached prime so the next
                # attempt re-resolves rather than fast-pathing a dead entry.
                self._primed.pop(peer_id, None)
                raise
            # Re-insert at the end (most-recently-primed) and evict the oldest over the cap.
            # Every live peer is re-primed each TTL window, so the front is the most-stale.
            self._primed.pop(peer_id, None)
            self._primed[peer_id] = time.monotonic() + max(0.0, ttl)
            while len(self._primed) > self._PRIMED_CACHE_MAX:
                self._primed.popitem(last=False)
            # Observability: re-resolution (force_refresh) is the rare/interesting event
            # (a stale or dropped peer recovering), so surface it at INFO; routine first
            # primes stay at DEBUG.
            log = logger.info if force_refresh else logger.debug
            log(f"Primed peerstore for {peer_id} from address book ({len(maddrs)} maddr(s), refresh={force_refresh})")

    async def _call_unary_protobuf_handler(
        self,
        peer_id: PeerID,
        handle_name: str,
        input: TInputProtobuf,
        output_protobuf_type: type[Message],
    ) -> Awaitable[TOutputProtobuf]:
        if self._peer_resolver is None:
            # Legacy path (daemon DHT resolves): unchanged.
            serialized_input = input.SerializeToString()
            response = await self._client.call_unary_handler(peer_id, handle_name, serialized_input)
            return output_protobuf_type.FromString(response)

        # Prime the peerstore first. We do NOT retry the unary call: it may have reached
        # the remote handler (duplicate-execution risk); recovery is owned by the caller's
        # retry/failover (e.g. BalancedRemoteExpert). On a connection-level failure we
        # invalidate the prime so the caller's next attempt re-resolves. A P2PHandlerError
        # (the remote handler ran) is NOT caught here -> the peer stays primed.
        await self._ensure_peer_connected(peer_id)
        serialized_input = input.SerializeToString()
        try:
            response = await self._client.call_unary_handler(peer_id, handle_name, serialized_input)
        except (P2PDaemonError, ControlFailure):
            self._primed.pop(peer_id, None)
            raise
        return output_protobuf_type.FromString(response)

    async def iterate_protobuf_handler(
        self,
        peer_id: PeerID,
        name: str,
        input: TInputProtobuf | TInputStream,
        output_protobuf_type: type[Message],
    ) -> TOutputStream:
        requests = input if isinstance(input, AsyncIterableABC) else as_aiter(input)
        return await self._iterate_protobuf_stream_handler(peer_id, name, requests, output_protobuf_type)

    def _start_listening(self) -> None:
        async def listen() -> None:
            async with self._client.listen():
                await asyncio.Future()  # Wait until this task will be cancelled in _terminate()

        self._listen_task = asyncio.create_task(listen())

    async def add_binary_stream_handler(
        self, name: str, handler: p2pclient.StreamHandler, balanced: bool = False
    ) -> None:
        if self._listen_task is None:
            self._start_listening()
        await self._client.stream_handler(name, handler, balanced)

    async def remove_binary_stream_handler(self, name: str) -> None:
        await self._client.remove_stream_handler(name)

    async def call_binary_stream_handler(
        self, peer_id: PeerID, handler_name: str
    ) -> tuple[StreamInfo, asyncio.StreamReader, asyncio.StreamWriter]:
        if self._peer_resolver is None:
            # Legacy path (daemon DHT resolves, no bounded dial timeout): unchanged.
            return await self._client.stream_open(peer_id, (handler_name,))

        # Resolver path: prime, then open the stream under a bounded dial timeout (stream
        # callers can't pass an RPC timeout, so the daemon's 60s default would otherwise
        # apply). Opening fails before any data reaches a handler (pre-handler), so on a
        # connection-level failure we drop the (possibly stale) prime, re-resolve fresh,
        # and retry the open exactly once.
        await self._ensure_peer_connected(peer_id)
        try:
            return await asyncio.wait_for(
                self._client.stream_open(peer_id, (handler_name,)), timeout=self._dial_timeout
            )
        except (P2PDaemonError, ControlFailure, asyncio.TimeoutError):
            self._primed.pop(peer_id, None)
            await self._ensure_peer_connected(peer_id, force_refresh=True)
            try:
                return await asyncio.wait_for(
                    self._client.stream_open(peer_id, (handler_name,)), timeout=self._dial_timeout
                )
            except (P2PDaemonError, ControlFailure, asyncio.TimeoutError):
                self._primed.pop(peer_id, None)  # don't keep a prime that just failed
                raise

    def __del__(self):
        self._terminate()

    @property
    def is_alive(self) -> bool:
        return self._alive

    async def shutdown(self) -> None:
        self._terminate()
        if self._child is not None:
            await self._child.wait()

    def _terminate(self) -> None:
        if self._client is not None:
            self._client.close()
        if self._listen_task is not None:
            cancel_task_if_running(self._listen_task)
        if self._reader_task is not None:
            cancel_task_if_running(self._reader_task)

        self._alive = False
        if self._child is not None and self._child.returncode is None:
            with suppress(ProcessLookupError):
                self._child.terminate()
                logger.debug(f"Terminated p2pd with id = {self.peer_id}")

            with suppress(FileNotFoundError, TypeError):
                os.remove(self._daemon_listen_maddr["unix"])
        with suppress(FileNotFoundError, TypeError):
            os.remove(self._client_listen_maddr["unix"])

    @staticmethod
    def _make_process_args(*args, **kwargs) -> list[str]:
        proc_args = []
        proc_args.extend(str(entry) for entry in args)
        proc_args.extend(
            f"-{key}={P2P._convert_process_arg_type(value)}" if value is not None else f"-{key}"
            for key, value in kwargs.items()
        )
        return proc_args

    @staticmethod
    def _convert_process_arg_type(val: Any) -> Any:
        if isinstance(val, bool):
            return int(val)
        return val

    @staticmethod
    def _maddrs_to_str(maddrs: list[Multiaddr]) -> str:
        return ",".join(str(addr) for addr in maddrs)

    async def _read_outputs(self, ready: asyncio.Future) -> None:
        last_line = None
        while True:
            line = await self._child.stdout.readline()
            if not line:  # Stream closed
                break
            last_line = line.rstrip().decode(errors="ignore")

            self._log_p2pd_message(last_line)
            if last_line.startswith("Peer ID:"):
                ready.set_result(None)

        if not ready.done():
            ready.set_exception(P2PDaemonError(f"Daemon failed to start: {last_line}"))

    @staticmethod
    def _log_p2pd_message(line: str) -> None:
        if '"logger"' not in line:  # User-friendly info from p2pd stdout
            logger.debug(line, extra={"caller": "p2pd"})
            return

        try:
            record = json.loads(line)
            caller = record["caller"]

            level = golog_level_to_python(record["level"])
            if level <= logging.WARNING:
                # Many Go loggers are excessively verbose (e.g. show warnings for unreachable peers),
                # so we downgrade INFO and WARNING messages to DEBUG.
                # The Go verbosity can still be controlled via the GOLOG_LOG_LEVEL env variable.
                # Details: https://github.com/ipfs/go-log#golog_log_level
                level = logging.DEBUG

            message = record["msg"]
            if "error" in record:
                message += f": {record['error']}"

            logger.log(
                level,
                message,
                extra={
                    "origin_created": datetime.strptime(record["ts"], "%Y-%m-%dT%H:%M:%S.%f%z").timestamp(),
                    "caller": caller,
                },
            )
        except Exception:
            # Parsing errors are unlikely, but we don't want to lose these messages anyway
            logger.warning(line, extra={"caller": "p2pd"})
            logger.exception("Failed to parse go-log message:")
