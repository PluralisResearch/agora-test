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
Originally taken from: https://github.com/mhchia/py-libp2p-daemon-bindings
Licence: MIT
Author: Kevin Mai-Husan Chia
"""

import asyncio

from google.protobuf.message import Message as PBMessage

from agora_server.hivemind.proto import p2pd_pb2 as p2pd_pb


DEFAULT_MAX_BITS: int = 64


class P2PHandlerError(Exception):
    """
    Raised if remote handled a request with an exception
    """


class P2PDaemonError(Exception):
    """
    Raised if daemon failed to handle request
    """


class ControlFailure(P2PDaemonError):
    pass


class DispatchFailure(P2PDaemonError):
    pass


async def write_unsigned_varint(stream: asyncio.StreamWriter, integer: int, max_bits: int = DEFAULT_MAX_BITS) -> None:
    max_int = 1 << max_bits
    if integer < 0:
        raise ValueError(f"negative integer: {integer}")
    if integer >= max_int:
        raise ValueError(f"integer too large: {integer}")
    while True:
        value = integer & 0x7F
        integer >>= 7
        if integer != 0:
            value |= 0x80
        byte = value.to_bytes(1, "big")
        stream.write(byte)
        await stream.drain()
        if integer == 0:
            break


async def read_unsigned_varint(stream: asyncio.StreamReader, max_bits: int = DEFAULT_MAX_BITS) -> int:
    max_int = 1 << max_bits
    iteration = 0
    result = 0
    has_next = True
    while has_next:
        data = await stream.readexactly(1)
        c = data[0]
        value = c & 0x7F
        result |= value << (iteration * 7)
        has_next = (c & 0x80) != 0
        iteration += 1
        if result >= max_int:
            raise ValueError(f"Varint overflowed: {result}")
    return result


def raise_if_failed(response: p2pd_pb.Response) -> None:
    if response.type == p2pd_pb.Response.ERROR:
        raise ControlFailure(f"Connect failed. msg={response.error.msg}")


async def write_pbmsg(stream: asyncio.StreamWriter, pbmsg: PBMessage) -> None:
    size = pbmsg.ByteSize()
    await write_unsigned_varint(stream, size)
    msg_bytes: bytes = pbmsg.SerializeToString()
    stream.write(msg_bytes)
    await stream.drain()


async def read_pbmsg_safe(stream: asyncio.StreamReader, pbmsg: PBMessage) -> None:
    len_msg_bytes = await read_unsigned_varint(stream)
    msg_bytes = await stream.readexactly(len_msg_bytes)
    pbmsg.ParseFromString(msg_bytes)
