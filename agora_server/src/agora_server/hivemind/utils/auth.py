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
import functools
import secrets

from abc import ABC, abstractmethod
from datetime import timedelta
from enum import Enum
from typing import Optional

from agora_server.hivemind.proto.auth_pb2 import AccessToken, RequestAuthInfo, ResponseAuthInfo
from agora_server.hivemind.utils.crypto import RSAPrivateKey, RSAPublicKey
from agora_server.hivemind.utils.logging import get_logger
from agora_server.hivemind.utils.timed_storage import TimedStorage, get_dht_time


logger = get_logger(__name__)


class AuthorizedRequestBase:
    """
    Interface for protobufs with the ``RequestAuthInfo auth`` field. Used for type annotations only.
    """

    auth: RequestAuthInfo


class AuthorizedResponseBase:
    """
    Interface for protobufs with the ``ResponseAuthInfo auth`` field. Used for type annotations only.
    """

    auth: ResponseAuthInfo


class AuthorizerBase(ABC):
    @abstractmethod
    async def sign_request(self, request: AuthorizedRequestBase, service_public_key: RSAPublicKey | None) -> None: ...

    @abstractmethod
    async def validate_request(self, request: AuthorizedRequestBase) -> bool: ...

    @abstractmethod
    async def sign_response(self, response: AuthorizedResponseBase, request: AuthorizedRequestBase) -> None: ...

    @abstractmethod
    async def validate_response(self, response: AuthorizedResponseBase, request: AuthorizedRequestBase) -> bool: ...


class TokenAuthorizerBase(AuthorizerBase):
    """
    Implements the authorization protocol for a moderated Hivemind network.
    See https://github.com/learning-at-home/hivemind/issues/253
    """

    def __init__(self, local_private_key: RSAPrivateKey | None = None):
        if local_private_key is None:
            local_private_key = RSAPrivateKey.process_wide()
        self._local_private_key = local_private_key
        self._local_public_key = local_private_key.get_public_key()

        self._local_access_token = None
        self._refresh_lock = asyncio.Lock()

        self._recent_nonces = TimedStorage()

    @abstractmethod
    async def get_token(self) -> AccessToken: ...

    @abstractmethod
    def is_token_valid(self, access_token: AccessToken) -> bool: ...

    @abstractmethod
    def does_token_need_refreshing(self, access_token: AccessToken) -> bool: ...

    async def refresh_token_if_needed(self) -> None:
        if self._local_access_token is None or self.does_token_need_refreshing(self._local_access_token):
            async with self._refresh_lock:
                if self._local_access_token is None or self.does_token_need_refreshing(self._local_access_token):
                    self._local_access_token = await self.get_token()
                    assert self.is_token_valid(self._local_access_token)

    @property
    def local_public_key(self) -> RSAPublicKey:
        return self._local_public_key

    @property
    def local_private_key(self) -> RSAPrivateKey:
        """The identity key this authorizer signs with. Exposed so call sites can
        thread the same named instance onward (e.g. into RedisDHT for envelope
        signing) instead of reading the process-wide singleton."""
        return self._local_private_key

    @property
    def local_access_token(self) -> AccessToken | None:
        """The most recently issued AccessToken (None before the first grant).
        Read by the Metadata Store token provider; refreshed via refresh_token /
        refresh_token_if_needed."""
        return self._local_access_token

    async def sign_request(self, request: AuthorizedRequestBase, service_public_key: RSAPublicKey | None) -> None:
        await self.refresh_token_if_needed()
        auth = request.auth

        auth.client_access_token.CopyFrom(self._local_access_token)

        if service_public_key is not None:
            auth.service_public_key = service_public_key.to_bytes()
        auth.time = get_dht_time()
        auth.nonce = secrets.token_bytes(8)

        assert auth.signature == b""
        auth.signature = self._local_private_key.sign(request.SerializeToString())

    _MAX_CLIENT_SERVICER_TIME_DIFF = timedelta(minutes=1)

    async def validate_request(self, request: AuthorizedRequestBase) -> bool:
        await self.refresh_token_if_needed()
        auth = request.auth

        if not self.is_token_valid(auth.client_access_token):
            logger.debug("Client failed to prove that it (still) has access to the network")
            return False

        client_public_key = RSAPublicKey.from_bytes(auth.client_access_token.public_key)
        signature = auth.signature
        auth.signature = b""
        if not client_public_key.verify(request.SerializeToString(), signature):
            logger.debug("Request has invalid signature")
            return False

        if auth.service_public_key and auth.service_public_key != self._local_public_key.to_bytes():
            logger.debug("Request is generated for a peer with another public key")
            return False

        with self._recent_nonces.freeze():
            current_time = get_dht_time()
            if abs(auth.time - current_time) > self._MAX_CLIENT_SERVICER_TIME_DIFF.total_seconds():
                logger.debug("Clocks are not synchronized or a previous request is replayed again")
                return False
            if auth.nonce in self._recent_nonces:
                logger.debug("Previous request is replayed again")
                return False

        self._recent_nonces.store(
            auth.nonce, None, current_time + self._MAX_CLIENT_SERVICER_TIME_DIFF.total_seconds() * 3
        )
        return True

    async def sign_response(self, response: AuthorizedResponseBase, request: AuthorizedRequestBase) -> None:
        await self.refresh_token_if_needed()
        auth = response.auth

        auth.service_access_token.CopyFrom(self._local_access_token)
        auth.nonce = request.auth.nonce

        assert auth.signature == b""
        auth.signature = self._local_private_key.sign(response.SerializeToString())

    async def validate_response(self, response: AuthorizedResponseBase, request: AuthorizedRequestBase) -> bool:
        await self.refresh_token_if_needed()
        auth = response.auth

        if not self.is_token_valid(auth.service_access_token):
            logger.debug("Service failed to prove that it (still) has access to the network")
            return False

        service_public_key = RSAPublicKey.from_bytes(auth.service_access_token.public_key)
        signature = auth.signature
        auth.signature = b""
        if not service_public_key.verify(response.SerializeToString(), signature):
            logger.debug("Response has invalid signature")
            return False

        if auth.nonce != request.auth.nonce:
            logger.debug("Response is generated for another request")
            return False

        return True


class AuthRole(Enum):
    CLIENT = 0
    SERVICER = 1


class AuthRPCWrapper:
    def __init__(
        self,
        stub,
        role: AuthRole,
        authorizer: AuthorizerBase | None,
        service_public_key: RSAPublicKey | None = None,
    ):
        self._stub = stub
        self._role = role
        self._authorizer = authorizer
        self._service_public_key = service_public_key

    def __getattribute__(self, name: str):
        if not name.startswith("rpc_"):
            return object.__getattribute__(self, name)

        method = getattr(self._stub, name)

        @functools.wraps(method)
        async def wrapped_rpc(request: AuthorizedRequestBase, *args, **kwargs):
            if self._authorizer is not None:
                if self._role == AuthRole.CLIENT:
                    await self._authorizer.sign_request(request, self._service_public_key)
                elif self._role == AuthRole.SERVICER:
                    if not await self._authorizer.validate_request(request):
                        return None

            response = await method(request, *args, **kwargs)

            if self._authorizer is not None:
                if self._role == AuthRole.SERVICER:
                    await self._authorizer.sign_response(response, request)
                elif self._role == AuthRole.CLIENT:
                    if not await self._authorizer.validate_response(response, request):
                        return None

            return response

        return wrapped_rpc
