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

from __future__ import annotations

import base64
import threading

from abc import ABC, abstractmethod

from cryptography import exceptions
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)


class PrivateKey(ABC):
    @abstractmethod
    def sign(self, data: bytes) -> bytes: ...

    @abstractmethod
    def get_public_key(self) -> PublicKey: ...


class PublicKey(ABC):
    @abstractmethod
    def verify(self, data: bytes, signature: bytes) -> bool: ...

    @abstractmethod
    def to_bytes(self) -> bytes: ...

    @classmethod
    @abstractmethod
    def from_bytes(cls, key: bytes) -> bytes: ...


_RSA_PADDING = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH)
_RSA_HASH_ALGORITHM = hashes.SHA256()


class RSAPrivateKey(PrivateKey):
    def __init__(self):
        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    @classmethod
    def from_der_bytes(cls, data: bytes) -> RSAPrivateKey:
        """Construct a named instance from DER-serialized private-key bytes (the
        ``to_bytes()`` format, i.e. the ``data`` field of the identity file's
        ``crypto_pb2.PrivateKey``). Does not touch the process-wide singleton."""
        key = cls.__new__(cls)
        key._private_key = serialization.load_der_private_key(data, password=None)
        return key

    _process_wide_key = None
    _process_wide_key_lock = threading.RLock()

    @classmethod
    def process_wide(cls) -> RSAPrivateKey:
        if cls._process_wide_key is None:
            with cls._process_wide_key_lock:
                if cls._process_wide_key is None:
                    cls._process_wide_key = cls()
        return cls._process_wide_key

    @classmethod
    def set_process_wide(cls, key: RSAPrivateKey) -> None:
        """Install ``key`` as the process-wide key (legacy DHT-mode components read it
        via the ``private_key=None`` fallbacks). Replaces the historical shim that
        mutated a lazily-created random key's internals. Metadata Store code paths
        never read the singleton — they take a named instance explicitly."""
        with cls._process_wide_key_lock:
            if cls._process_wide_key is not None and cls._process_wide_key.to_bytes() != key.to_bytes():
                # Anything constructed before this point captured the old (random) key
                # and its public-key tag; that ordering was equally broken under the
                # old shim, but make it visible instead of silent.
                logger.warning(
                    "Replacing an already-initialized process-wide RSA key; components "
                    "constructed earlier keep the previous key"
                )
            cls._process_wide_key = key

    def sign(self, data: bytes) -> bytes:
        signature = self._private_key.sign(data, _RSA_PADDING, _RSA_HASH_ALGORITHM)
        return base64.b64encode(signature)

    def get_public_key(self) -> RSAPublicKey:
        return RSAPublicKey(self._private_key.public_key())

    def to_bytes(self) -> bytes:
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Serializes the private key to make the class instances picklable
        state["_private_key"] = self.to_bytes()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._private_key = serialization.load_der_private_key(self._private_key, password=None)


class RSAPublicKey(PublicKey):
    def __init__(self, public_key: rsa.RSAPublicKey):
        self._public_key = public_key

    def verify(self, data: bytes, signature: bytes) -> bool:
        try:
            signature = base64.b64decode(signature)

            # Returns None if the signature is correct, raises an exception otherwise
            self._public_key.verify(signature, data, _RSA_PADDING, _RSA_HASH_ALGORITHM)

            return True
        except (ValueError, exceptions.InvalidSignature):
            return False

    def to_bytes(self) -> bytes:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.OpenSSH, format=serialization.PublicFormat.OpenSSH
        )

    @classmethod
    def from_bytes(cls, key: bytes) -> RSAPublicKey:
        key = serialization.load_ssh_public_key(key)
        if not isinstance(key, rsa.RSAPublicKey):
            raise ValueError(f"Expected an RSA public key, got {key}")
        return cls(key)
