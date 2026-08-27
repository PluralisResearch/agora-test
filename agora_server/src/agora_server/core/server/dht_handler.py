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

import threading

from collections.abc import Sequence
from functools import partial
from typing import NamedTuple

from agora_server.core.address_book import peer_address_book_key
from agora_server.hivemind.dht import DHT, DHTNode, DHTValue
from agora_server.hivemind.dht.crypto import RSASignatureValidator
from agora_server.hivemind.moe.client.expert import RemoteExpert, create_remote_experts
from agora_server.hivemind.moe.expert_uid import (
    FLAT_EXPERT,
    UID_DELIMITER,
    UID_PATTERN,
    Coordinate,
    ExpertPrefix,
    ExpertUID,
    is_valid_uid,
    split_uid,
)
from agora_server.hivemind.p2p import PeerID
from agora_server.hivemind.utils import (
    MAX_DHT_TIME_DISCREPANCY_SECONDS,
    DHTExpiration,
    MPFuture,
    ValueWithExpiration,
    get_dht_time,
    get_logger,
)
from agora_server.types import GhostPhase


logger = get_logger(__name__)


_OWNER_SUBKEY: bytes | None = None


def _owner_subkey(dht: DHT) -> bytes:
    """The local peer's owner marker -- b"[owner:<rsa pubkey>]" -- used to tag expert records.

    Metadata Store mode: sourced from the RedisDHT handle (the marker of the key its envelopes
    are signed with), so the tag can never diverge from the envelope pubkey
    the store verifies against; this also covers unmanaged runs, where no authorization call
    initializes the process-wide key. DHT mode (or an unsigned RedisDHT, whose property is None):
    the process-wide RSA key, i.e. the identity installed by `set_process_wide` during
    authorization -- the registered RSASignatureValidator signs/verifies the tagged records.
    That fallback is cached after first use: the process-wide key is fixed once the peer has
    authorized, which always precedes declaration.
    """
    dht_owner_tag = getattr(dht, "local_public_key", None)
    if dht_owner_tag is not None:
        return dht_owner_tag
    global _OWNER_SUBKEY
    if _OWNER_SUBKEY is None:
        _OWNER_SUBKEY = RSASignatureValidator().local_public_key
    return _OWNER_SUBKEY


def _prefix_subkey(coordinate: Coordinate, owner: bytes) -> bytes:
    """Subkey for an expert PREFIX record: the coordinate (preserves per-expert uniqueness when one
    peer hosts several experts under the same prefix) followed by the owner marker (so the record is
    owner-protected). Readers ignore the subkey, and the `[owner:...]` marker is matched by substring,
    so the leading coordinate is harmless."""
    return f"{coordinate}".encode() + b":" + owner


def extract_leaf_record(record: ValueWithExpiration | None) -> ValueWithExpiration | None:
    """Resolve an expert LEAF lookup (``node.get_many()[uid]``) to a single ValueWithExpiration whose
    ``.value`` is the expert_dht_value string.

    Expert leaves are owner-tagged (subkey = ``[owner:<pubkey>]``), so a leaf is a Pattern-B record:
    ``found[uid].value`` is a dict ``{owner_subkey -> ValueWithExpiration(str)}``. There is exactly one
    owner per uid; we defensively take the freshest entry. Also accepts the legacy Pattern-A shape (a
    bare ``str`` value) for rolling compatibility during rollout.

    NOTE: this is rolling-compatible by design and does NOT *enforce* ownership -- it ignores the
    subkey and will accept an unowned (no ``[owner:]``) or legacy Pattern-A entry if it is the
    freshest, so a not-yet-upgraded worker's expert is still discoverable mid-rollout. Rejecting
    unowned/unauthorized records (a require-owner SchemaValidator + owner authorization) is not
    done here; in Metadata Store mode the store enforces it server-side.
    """
    if record is None:
        return None
    if isinstance(record.value, str):
        return record  # legacy Pattern A (pre-owner-tag) -- rolling compatibility
    if isinstance(record.value, dict):
        candidates = [
            item
            for item in record.value.values()
            if isinstance(item, ValueWithExpiration) and isinstance(item.value, str)
        ]
        return max(candidates, key=lambda item: item.expiration_time) if candidates else None
    return None


class ExpertInfo(NamedTuple):
    """
    Information about a single expert in the DHT: its uid, the peer that hosts it, and its ghost phase.
    Phase 1 ghost experts are not sent any batches from trainers, but receive AR updates.
    Phase 2 ghost experts receive batches (for optimizer warm-up) but don't contribute to AR.
    ghost_phase_start_epoch is the epoch at which the current ghost phase began (None when ghost_phase is OFF).
    """

    uid: ExpertUID
    peer_id: PeerID
    ghost_phase: GhostPhase = GhostPhase.OFF
    ghost_phase_start_epoch: int | None = None


class DHTHandlerThread(threading.Thread):
    def __init__(
        self,
        module_backends,
        dht: DHT,
        update_period: float = 30,
        expiration: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if expiration is None:
            expiration = max(2 * update_period, MAX_DHT_TIME_DISCREPANCY_SECONDS)
        self.module_backends = module_backends
        self.dht = dht
        self.update_period = update_period
        self.expiration = expiration
        self.stop = threading.Event()

    def run(self) -> None:
        declare_experts(
            self.dht,
            self.module_backends.keys(),
            expiration_time=get_dht_time() + self.expiration,
            ghost_phases=[backend.optimizer.ghost_phase for backend in self.module_backends.values()],
            ghost_phase_start_epochs=[
                backend.optimizer.ghost_phase_start_epoch for backend in self.module_backends.values()
            ],
        )
        while not self.stop.wait(self.update_period):
            declare_experts(
                self.dht,
                self.module_backends.keys(),
                expiration_time=get_dht_time() + self.expiration,
                ghost_phases=[backend.optimizer.ghost_phase for backend in self.module_backends.values()],
                ghost_phase_start_epochs=[
                    backend.optimizer.ghost_phase_start_epoch for backend in self.module_backends.values()
                ],
            )


def declare_experts(
    dht: DHT,
    uids: Sequence[ExpertUID],
    expiration_time: DHTExpiration,
    ghost_phases: Sequence[GhostPhase] | None = None,
    ghost_phase_start_epochs: Sequence[int | None] | None = None,
    wait: bool = True,
) -> dict[ExpertUID, bool] | MPFuture[dict[ExpertUID, bool]]:
    """Make experts visible to all DHT peers; update timestamps if declared previously.

    Args:
        uids (Sequence[ExpertUID]): A list of expert ids to update.
        expiration_time (DHTExpiration): Experts will be visible for this many seconds.
        ghost_phases (Sequence[GhostPhase]|None, optional): If specified, must be of the same length as uids. Indicates ghost phase for each expert. Defaults to None (OFF for all).
        ghost_phase_start_epochs (Sequence[int|None]|None, optional): If specified, must be of the same length as uids. Indicates the epoch at which each expert's current ghost phase began. Required (and used) only when the corresponding ghost_phase is not OFF. Defaults to None.
        wait (bool, optional): If True, awaits for declaration to finish, otherwise runs in background. Defaults to True.
    Returns:
        dict[ExpertUID, bool] | MPFuture[dict[ExpertUID, bool]]: If wait, returns store status for every key (True = store succeeded, False = store rejected).
    """
    assert not isinstance(uids, str), "Please send a list / tuple of expert uids."
    if not isinstance(uids, list):
        uids = list(uids)
    for uid in uids:
        assert is_valid_uid(uid), f"{uid} is not a valid expert uid. All uids must follow {UID_PATTERN.pattern}"
    if ghost_phases is not None:
        assert len(ghost_phases) == len(uids), "If ghost_phases is specified, it must be of the same length as uids."
    else:
        ghost_phases = [GhostPhase.OFF] * len(uids)
    if ghost_phase_start_epochs is not None:
        assert len(ghost_phase_start_epochs) == len(uids), (
            "If ghost_phase_start_epochs is specified, it must be of the same length as uids."
        )
    else:
        ghost_phase_start_epochs = [None] * len(uids)
    return dht.run_coroutine(
        partial(
            _declare_experts,
            uids=uids,
            expiration_time=expiration_time,
            ghost_phases=ghost_phases,
            ghost_phase_start_epochs=ghost_phase_start_epochs,
        ),
        return_future=not wait,
    )


async def _declare_experts(
    dht: DHT,
    node: DHTNode,
    uids: list[ExpertUID],
    ghost_phases: list[GhostPhase],
    ghost_phase_start_epochs: list[int | None],
    expiration_time: DHTExpiration,
) -> dict[ExpertUID, bool]:
    # We always store at least the peer address-book record below, so floor workers at 1:
    # a real DHTNode.store_many with num_workers=0 hits asyncio.wait([]) (e.g. empty uids).
    num_workers = max(1, len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers))
    data_to_store: dict[tuple[ExpertPrefix, Coordinate | None], DHTValue] = {}

    for uid, ghost_phase, ghost_phase_start_epoch in zip(uids, ghost_phases, ghost_phase_start_epochs):
        expert_dht_value = format_expert_dht_value(dht.peer_id, ghost_phase, ghost_phase_start_epoch)
        # Owner-tag expert records: put the writer's identity in the SUBKEY (not just the value), so
        # in DHT mode the registered RSASignatureValidator signs+enforces them, and in Metadata Store
        # mode the marker is carried for server-side ownership verification. Values are unchanged.
        # The leaf moves Pattern A -> Pattern B (subkey = owner marker); the prefix subkey keeps the
        # coordinate (uniqueness) and appends the owner marker.
        owner = _owner_subkey(dht)
        data_to_store[uid, owner] = expert_dht_value
        prefix = uid if uid.count(UID_DELIMITER) > 1 else f"{uid}{UID_DELIMITER}{FLAT_EXPERT}"
        for _ in range(prefix.count(UID_DELIMITER) - 1):
            prefix, last_coord = split_uid(prefix)
            data_to_store[prefix, _prefix_subkey(last_coord, owner)] = (uid, expert_dht_value)

    # Co-publish this node's address-book entry (peer_id -> visible multiaddrs) in the same
    # write, so peers can resolve it without the DHT (M4). Same key/expiration/refresh cadence
    # as the expert records. Where our own maddrs come from depends on the node:
    #   - DHT mode: the real DHTNode exposes `.p2p` bound to this callback's loop -- read it here.
    #   - Metadata Store mode: RedisNode has no `.p2p`, but RedisDHT.get_visible_maddrs() is a
    #     cached synchronous read (SwarmP2P) that does NOT re-enter run_coroutine, so calling it
    #     from inside the callback is safe (unlike the real DHT's get_visible_maddrs()).
    # Precondition: a RedisDHT used to DECLARE must wrap a sync get_visible_maddrs source
    # (SwarmP2P) -- a RedisDHT wrapping a real DHT would re-enter run_coroutine here.
    # Maddrs are stored verbatim (with the /p2p/<id> suffix); the resolver decapsulates.
    # See dht-redis-impl-m3-m4.md s4.1.
    node_p2p = getattr(node, "p2p", None)
    if node_p2p is not None:
        visible_maddrs = [str(maddr) for maddr in await node_p2p.get_visible_maddrs()]
    else:
        visible_maddrs = [str(maddr) for maddr in dht.get_visible_maddrs()]
    data_to_store[peer_address_book_key(dht.peer_id), None] = visible_maddrs

    keys, maybe_subkeys, values = zip(*((key, subkey, value) for (key, subkey), value in data_to_store.items()))
    store_ok = await node.store_many(keys, values, expiration_time, subkeys=maybe_subkeys, num_workers=num_workers)
    return store_ok


def get_experts(
    dht: DHT,
    uids: list[ExpertUID],
    expiration_time: DHTExpiration | None = None,
    return_future: bool = False,
) -> list[RemoteExpert | None] | MPFuture[list[RemoteExpert | None]]:
    """Get RemoteExpert instances for experts with given uids from the DHT.

    Args:
        dht (DHT): DHT instance to use for lookup.
        uids (list[ExpertUID]): Find experts with these ids from across the DHT.
        expiration_time (DHTExpiration | None, optional): If specified, return experts that expire no sooner than this (based on get_dht_time). Defaults to None.
        return_future (bool, optional): If False (default), return when finished. Otherwise return MPFuture and run in background. Defaults to False.

    Returns:
        list[RemoteExpert|None]|MPFuture[list[RemoteExpert|None]]: a list of [RemoteExpert if found else None].
    """
    assert not isinstance(uids, str), "Please send a list / tuple of expert uids."
    result = dht.run_coroutine(partial(_get_experts, uids=list(uids), expiration_time=expiration_time), return_future)
    return create_remote_experts(result, dht, return_future)


async def _get_experts(
    dht: DHT,
    node: DHTNode,
    uids: list[ExpertUID],
    expiration_time: DHTExpiration | None,
) -> list[ExpertInfo | None]:
    if expiration_time is None:
        expiration_time = get_dht_time()
    num_workers = len(uids) if dht.num_workers is None else min(len(uids), dht.num_workers)
    found: dict[ExpertUID, DHTValue] = await node.get_many(uids, expiration_time, num_workers=num_workers)

    experts: list[ExpertInfo | None] = [None] * len(uids)
    for i, uid in enumerate(uids):
        leaf = extract_leaf_record(found[uid])
        if leaf is not None and isinstance(leaf.value, str):
            peer_id, ghost_phase, ghost_phase_start_epoch = parse_expert_dht_value(leaf.value)
            experts[i] = ExpertInfo(uid, peer_id, ghost_phase, ghost_phase_start_epoch)
    return experts


def format_expert_dht_value(
    peer_id: PeerID,
    ghost_phase: GhostPhase = GhostPhase.OFF,
    ghost_phase_start_epoch: int | None = None,
) -> str:
    base = peer_id.to_base58()
    if ghost_phase == GhostPhase.PHASE1:
        return f"{base}_ghost1_{ghost_phase_start_epoch}"
    elif ghost_phase == GhostPhase.PHASE2:
        return f"{base}_ghost2_{ghost_phase_start_epoch}"
    return base


def parse_expert_dht_value(dht_value: str) -> tuple[PeerID, GhostPhase, int | None]:
    for suffix, phase in (("_ghost1_", GhostPhase.PHASE1), ("_ghost2_", GhostPhase.PHASE2)):
        idx = dht_value.rfind(suffix)
        if idx != -1:
            return (
                PeerID.from_base58(dht_value[:idx]),
                phase,
                int(dht_value[idx + len(suffix) :]),
            )
    return PeerID.from_base58(dht_value), GhostPhase.OFF, None
