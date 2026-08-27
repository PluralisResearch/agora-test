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
"""Self-describing per-microbatch metadata shared by client and server."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from agora_server.hivemind.utils.serializer import MSGPackSerializer


BATCH_METADATA_VERSION = 1


def _as_str_or_none(value: object) -> str | None:
    """Coerce a deserialized next-hop field to ``str`` or ``None``.

    The next-hop keys are optional and arrive over the network, so a malformed value (wrong
    type) is treated as absent rather than propagated past the parse boundary -- keeping the
    ``Optional[str]`` field contract intact for downstream consumers.
    """
    return value if isinstance(value, str) else None


def _as_manifest_or_none(value: object) -> str | None:
    if isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value):
        return value
    return None


def _as_maddrs_or_none(value: object) -> tuple[str, ...] | None:
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return tuple(value) or None
    return None


def _as_nonnegative_int_or_none(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _as_positive_int_or_none(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


class BatchPurpose(IntEnum):
    # Role of a microbatch -- NOT its direction. Forward vs backward is conveyed by the
    # RPC channel (rpc_forward vs rpc_backward), so it is not encoded here. NORMAL is
    # ordinary training compute; this is the extension point for future fault-tolerance
    # roles such as REDUNDANT_CACHE / REPLICATE
    NORMAL = 0


@dataclass(frozen=True)
class BatchMetadata:
    trainer_uid: str  # the trainer's UID; stable + unique per trainer
    seq: int  # per-trainer-instance monotonic microbatch counter
    data_shard: int = -1  # the trainer's dataset shard (data_idx)
    purpose: BatchPurpose = BatchPurpose.NORMAL
    # Peer id for the trainer-side w2w coordinator. In direct mode workers use this to report
    # accepted/done/drop events and to request replacement hops.
    trainer_peer_id: str | None = None
    # Optional next-hop coordination: the next-stage worker that should receive
    # this microbatch's activation.
    next_hop_uid: str | None = None
    next_hop_peer_id: str | None = None
    # Optional prev-hop coordination (the reverse hop): the predecessor / reverse hop --
    # the previous-stage worker.
    prev_hop_uid: str | None = None
    prev_hop_peer_id: str | None = None
    prev_hop_maddrs: tuple[str, ...] | None = None
    data_manifest: str | None = None
    sample_start: int | None = None
    sample_stride: int | None = None
    sample_rows: int | None = None

    def to_bytes(self) -> bytes:
        obj = {
            "v": BATCH_METADATA_VERSION,
            "tid": self.trainer_uid,
            "seq": self.seq,
            "shard": self.data_shard,
            "kind": int(self.purpose),
        }
        # Emit the next-hop keys ONLY when set, so legacy + tail bytes are byte-identical.
        if self.next_hop_uid is not None:
            obj["nh_uid"] = self.next_hop_uid
        if self.next_hop_peer_id is not None:
            obj["nh_pid"] = self.next_hop_peer_id
        if self.trainer_peer_id is not None:
            obj["tpid"] = self.trainer_peer_id
        # Emit the prev-hop keys ONLY when set, so legacy + head bytes are byte-identical.
        if self.prev_hop_uid is not None:
            obj["ph_uid"] = self.prev_hop_uid
        if self.prev_hop_peer_id is not None:
            obj["ph_pid"] = self.prev_hop_peer_id
        if self.prev_hop_maddrs is not None:
            obj["ph_maddrs"] = list(self.prev_hop_maddrs)
        if self.data_manifest is not None:
            obj["manifest"] = self.data_manifest
        if self.sample_start is not None:
            obj["sample_start"] = self.sample_start
        if self.sample_stride is not None:
            obj["sample_stride"] = self.sample_stride
        if self.sample_rows is not None:
            obj["sample_rows"] = self.sample_rows
        return MSGPackSerializer.dumps(obj)

    @classmethod
    def from_bytes(cls, data: bytes) -> Optional["BatchMetadata"]:
        if not data:
            return None
        try:
            obj = MSGPackSerializer.loads(data)
            if not isinstance(obj, dict):
                return None
            return cls(
                trainer_uid=obj["tid"],
                seq=obj["seq"],
                data_shard=obj.get("shard", -1),
                purpose=BatchPurpose(obj.get("kind", 0)),
                trainer_peer_id=_as_str_or_none(obj.get("tpid")),
                next_hop_uid=_as_str_or_none(obj.get("nh_uid")),
                next_hop_peer_id=_as_str_or_none(obj.get("nh_pid")),
                prev_hop_uid=_as_str_or_none(obj.get("ph_uid")),
                prev_hop_peer_id=_as_str_or_none(obj.get("ph_pid")),
                prev_hop_maddrs=_as_maddrs_or_none(obj.get("ph_maddrs")),
                data_manifest=_as_manifest_or_none(obj.get("manifest")),
                sample_start=_as_nonnegative_int_or_none(obj.get("sample_start")),
                sample_stride=_as_positive_int_or_none(obj.get("sample_stride")),
                sample_rows=_as_positive_int_or_none(obj.get("sample_rows")),
            )
        except Exception:
            return None
