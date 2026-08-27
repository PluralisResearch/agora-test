# Copyright 2026 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Shared Metadata Store address-book format (peer_id -> multiaddrs).

Single source of truth for the address-book **key format** and the
stored-maddr -> dialable-transport-maddr conversion, used by both the writer
(`_declare_experts` in `core/server/dht_handler.py`) and the reader (the P2P
peer resolver, `core/peer_resolver.py`). Kept dependency-light so it can be
imported wherever a P2P transport lives.

A peer's record is a Pattern-A entry: `peer:{peer_id_base58}` -> `[str(maddr)]`,
where each maddr is exactly what `P2P.get_visible_maddrs()` reported (i.e. with
the `/p2p/<peer_id>` suffix encapsulated). The resolver strips that suffix before
`connect()` (which takes the peer_id separately) and validates it matches the
target, so a record can never redirect a dial to a different peer.
"""

from __future__ import annotations

from agora_server.hivemind.p2p import PeerID
from agora_server.hivemind.utils.multiaddr import Multiaddr


PEER_ADDRESS_BOOK_PREFIX = "peer:"


def peer_address_book_key(peer_id: PeerID) -> str:
    """Metadata Store key for a peer's address-book record (Pattern A: peer_id -> [multiaddrs])."""
    return f"{PEER_ADDRESS_BOOK_PREFIX}{peer_id.to_base58()}"


def transport_maddrs(maddr_strings: list[str], peer_id: PeerID) -> list[Multiaddr]:
    """Convert stored address-book maddr strings into transport multiaddrs for ``connect``.

    Strips the trailing ``/p2p/<id>`` (``connect`` takes the peer_id separately) and
    keeps only addresses whose embedded id matches ``peer_id`` — a record must not be
    able to redirect a dial to a different peer. A string without a ``/p2p/`` segment is
    kept as-is. (Relay/circuit addresses are out of scope until M5.)
    """
    expected = peer_id.to_base58()
    out: list[Multiaddr] = []
    for s in maddr_strings:
        if "/p2p/" in s:
            head, _, tail = s.partition("/p2p/")
            embedded = tail.split("/", 1)[0]
            if embedded != expected:
                continue  # record points at a different peer_id -> drop it
            s = head
        out.append(Multiaddr(s))
    return out
