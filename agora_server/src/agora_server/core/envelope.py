# Copyright 2026 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Client-side envelope for records written through the Metadata Store.

An envelope wraps the caller's value with metadata the Metadata Store needs:

    envelope = msgpack({
        payload:    msgpack(value),     # the caller's value, opaque to the store
        expiration: <absolute unix float>,
        pubkey:     <OpenSSH RSA public key bytes>,      # when signed
        signature:  <RSAPrivateKey.sign(signed_bytes)>,  # when signed
    })

The signature covers ``msgpack((str(key), subkey, payload, expiration, pubkey))``
— the record's full binding, so a signed envelope cannot be replayed under a
different key/subkey or with a stretched expiration. The store treats
``payload`` as opaque and stores the whole envelope verbatim (which is what
makes byte-exact verification possible on both sides). The server verifies
with a byte-identical copy of :func:`signed_bytes` in
``metadata_store/.../envelope.py`` (the slim service does not import
agora_server).

Read policy: a record with **no** ``pubkey``/``signature`` (not-yet-upgraded
writer) is accepted with a rate-limited warning; a record that carries them
but fails verification raises :class:`EnvelopeVerificationError` and is
dropped by callers. Ownership binding on read checks every
``[owner:<pubkey>]`` tag in the key/subkey against the envelope pubkey; the
peer-id form runs only where the caller knows the key names a peer
(``expected_owner_peer_id``, used by the peer resolver).
"""

import re
import time

from typing import Any, cast

import msgpack

from agora_server.hivemind.p2p import PeerID
from agora_server.hivemind.utils.crypto import RSAPrivateKey, RSAPublicKey
from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)

# The owner-tag wire format is hivemind's RSASignatureValidator convention
# (b"[owner:<ssh-rsa ...>]"), shared with the DHT-mode validators and the
# Metadata Store's server-side ownership check.
OWNER_TAG_FORMAT = b"[owner:_key_]"
_OWNER_TAG_RE = re.compile(re.escape(OWNER_TAG_FORMAT).replace(b"_key_", rb"(.+?)"))

# Rate-limit for the accepted-unsigned-record warning: unsigned records are
# routine while old writers remain in the swarm, so warn with a count, at most
# once per window. Lock-free on purpose — a racy duplicate warning is fine.
_UNSIGNED_WARN_INTERVAL_S = 60.0
_unsigned_seen_count = 0
_unsigned_last_warn = 0.0


class EnvelopeVerificationError(ValueError):
    """A signed envelope failed verification (bad signature, malformed pubkey,
    owner-tag mismatch, or peer-id binding mismatch). Callers drop the record."""


def _pack(value: Any) -> bytes:
    return cast(bytes, msgpack.dumps(value, use_bin_type=True))


def _unpack(data: bytes) -> Any:
    return msgpack.loads(data, raw=False)


def owner_tag(pubkey: bytes) -> bytes:
    """Format a serialized public key as the b"[owner:<pubkey>]" marker used to
    owner-tag keys/subkeys."""
    return OWNER_TAG_FORMAT.replace(b"_key_", pubkey)


def signed_bytes(key: Any, subkey: Any, payload: bytes, expiration: float, pubkey: bytes) -> bytes:
    """The canonical byte string an envelope signature covers.

    Both sides can reconstruct it exactly: ``key`` is coerced to ``str`` (the
    wire contract), ``subkey`` round-trips type-faithfully (str/int/bytes/None),
    and ``payload``/``expiration``/``pubkey``
    are taken verbatim from the envelope itself. The Metadata Store holds a
    server-side duplicate of this function — keep the two byte-identical.
    """
    return _pack([str(key), subkey, payload, float(expiration), pubkey])


def build(
    value: Any,
    expiration_time: float,
    *,
    key: Any = None,
    subkey: Any = None,
    private_key: RSAPrivateKey | None = None,
) -> bytes:
    """Wrap a caller value into an envelope, signed when ``private_key`` is given.

    An unsigned envelope is built when ``private_key`` is None — the
    read-only/ops paths; production writers thread their identity key.
    Signing requires the ``key`` context (the signature binds the record to it).
    """
    payload = _pack(value)
    envelope: dict[str, Any] = {"payload": payload, "expiration": float(expiration_time)}
    if private_key is not None:
        if key is None:
            raise ValueError("signing an envelope requires the record's key")
        pubkey = private_key.get_public_key().to_bytes()
        envelope["pubkey"] = pubkey
        envelope["signature"] = private_key.sign(signed_bytes(key, subkey, payload, expiration_time, pubkey))
    return _pack(envelope)


def read(
    envelope: bytes,
    *,
    key: Any = None,
    subkey: Any = None,
    expected_owner_peer_id: PeerID | None = None,
    require_signed: bool = False,
) -> tuple[Any, float]:
    """Unwrap an envelope into ``(value, expiration_time)``, verifying its signature.

    Raises :class:`EnvelopeVerificationError` when the envelope carries
    ``pubkey``/``signature`` and any check fails:

    - the signature does not verify over :func:`signed_bytes` recomputed from
      the caller's ``key``/``subkey`` context and the envelope's own fields;
    - an ``[owner:<pubkey>]`` tag in ``key``/``subkey`` differs from the
      envelope pubkey;
    - ``expected_owner_peer_id`` is given and differs from the peer_id derived
      from the envelope pubkey.

    NOTE: An envelope without those fields is accepted, including when ``expected_owner_peer_id`` is set —
    UNLESS ``require_signed`` is set; it is what makes forged-unsigned records from a compromised store detectable).
    (see also the MS server-side config: ``signature_enforcement=enforce``)
    Malformed msgpack raises whatever the decode raises, as before.
    """
    obj = _unpack(envelope)
    payload, expiration = obj["payload"], float(obj["expiration"])
    pubkey, signature = obj.get("pubkey"), obj.get("signature")

    if pubkey is None or signature is None:
        if require_signed:
            raise EnvelopeVerificationError(f"unsigned record for key {key!r} (require_signed_reads is on)")
        _note_unsigned(key)
        return _unpack(payload), expiration

    try:
        public_key = RSAPublicKey.from_bytes(pubkey)
    except Exception as exc:
        raise EnvelopeVerificationError(f"envelope for key {key!r} has an unloadable pubkey: {exc}") from exc
    if not public_key.verify(signed_bytes(key, subkey, payload, expiration, pubkey), signature):
        raise EnvelopeVerificationError(f"envelope signature verification failed for key {key!r}")

    _check_owner_tags(key, subkey, pubkey)

    if expected_owner_peer_id is not None:
        try:
            derived = PeerID.from_rsa_pubkey(pubkey)
        except Exception as exc:
            raise EnvelopeVerificationError(f"cannot derive peer_id from envelope pubkey for {key!r}: {exc}") from exc
        if derived != expected_owner_peer_id:
            raise EnvelopeVerificationError(
                f"envelope pubkey derives peer_id {derived} but the record names {expected_owner_peer_id}"
            )

    return _unpack(payload), expiration


def _check_owner_tags(key: Any, subkey: Any, pubkey: bytes) -> None:
    """Every [owner:...] tag present in the key/subkey must equal the envelope
    pubkey (mirrors RSASignatureValidator's rule that a record may not carry
    two different owners)."""
    tags = []
    for part in (key, subkey):
        if isinstance(part, bytes):
            tags += _OWNER_TAG_RE.findall(part)
        elif isinstance(part, str):
            tags += _OWNER_TAG_RE.findall(part.encode())
    for tag in tags:
        if tag != pubkey:
            raise EnvelopeVerificationError(f"envelope pubkey does not match the [owner:...] tag on key {key!r}")


def _note_unsigned(key: Any) -> None:
    global _unsigned_seen_count, _unsigned_last_warn
    _unsigned_seen_count += 1
    now = time.monotonic()
    if now - _unsigned_last_warn >= _UNSIGNED_WARN_INTERVAL_S:
        _unsigned_last_warn = now
        logger.warning(
            f"Accepted unsigned Metadata Store record for key {key!r} "
            f"({_unsigned_seen_count} unsigned reads so far; tolerated until signing is enforced)"
        )
