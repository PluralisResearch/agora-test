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

from collections.abc import Iterable, Sequence
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from torch.autograd.function import once_differentiable

from agora_server.hivemind import moe
from agora_server.hivemind.compression import (
    deserialize_tensor_stream,
    deserialize_torch_tensor,
    serialize_torch_tensor,
)
from agora_server.hivemind.dht import DHT
from agora_server.hivemind.moe.client.remote_expert_worker import RemoteExpertWorker
from agora_server.hivemind.moe.expert_uid import ExpertInfo
from agora_server.hivemind.p2p import P2P, PeerID, StubBase
from agora_server.hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, MAX_UNARY_PAYLOAD_SIZE
from agora_server.hivemind.proto import runtime_pb2
from agora_server.hivemind.utils.asyncio import amap_in_executor, iter_as_aiter
from agora_server.hivemind.utils.mpfuture import MPFuture
from agora_server.hivemind.utils.nested import nested_compare, nested_flatten, nested_pack
from agora_server.hivemind.utils.serializer import MSGPackSerializer
from agora_server.hivemind.utils.streaming import split_for_streaming


DUMMY = torch.empty(0, requires_grad=True)  # dummy tensor that triggers autograd in RemoteExpert


def get_server_stub(p2p: P2P, server_peer_id: PeerID) -> ConnectionHandlerStub:  # noqa: F821
    """Create an RPC stub that can send requests to any expert on the specified remote server"""
    return moe.server.connection_handler.ConnectionHandler.get_stub(p2p, server_peer_id)


class RemoteExpert(nn.Module):
    """
    A simple module that runs forward/backward of an expert hosted on a remote machine.
    Works seamlessly with pytorch autograd. (this is essentially a simple RPC function)
    Warning: RemoteExpert currently assumes that you provide it with correct input shapes.
    Sending wrong input shapes can cause RemoteExpert to freeze indefinitely due to error in runtime.

    :param expert_info: RemoteExpertInfo with uid and server PeerInfo
    :param p2p: P2P instance connected to the running p2pd
    """

    def __init__(self, expert_info: ExpertInfo, p2p: P2P):
        super().__init__()
        self._info, self.p2p = expert_info, p2p
        self._rpc_info = None

    @property
    def uid(self):
        return self._info.uid

    @property
    def peer_id(self) -> PeerID:
        return self._info.peer_id

    @property
    def stub(self) -> StubBase:
        return get_server_stub(self.p2p, self.peer_id)

    def forward(self, *args, **kwargs):
        """Call RemoteExpert for the specified inputs and return its output(s). Compatible with pytorch.autograd."""
        assert len(kwargs) == len(self.info["keyword_names"]), f"Keyword args should be {self.info['keyword_names']}"
        kwargs = {key: kwargs[key] for key in self.info["keyword_names"]}

        # Note: we put keyword arguments in the same order as on a server to prevent f(a=1, b=2) != f(b=2, a=1) errors

        forward_inputs = (args, kwargs)

        if not nested_compare(forward_inputs, self.info["forward_schema"]):
            raise TypeError("Inputs do not match expert input schema. Did you pass the right number of parameters?")

        flat_outputs = _RemoteModuleCall.apply(DUMMY, self.uid, self.stub, self.info, *nested_flatten(forward_inputs))

        # Note: we send DUMMY to prevent torch from excluding expert from backward if no other inputs require grad
        return nested_pack(flat_outputs, structure=self.info["outputs_schema"])

    @property
    def info(self):
        if self._rpc_info is None:
            outputs = RemoteExpertWorker.run_coroutine(self.stub.rpc_info(runtime_pb2.ExpertUID(uid=self.uid)))
            self._rpc_info = MSGPackSerializer.loads(outputs.serialized_info)
        return self._rpc_info

    def extra_repr(self):
        return f"uid={self.uid}, server_peer_id={self.peer_id}"


def _create_remote_experts(infos: Sequence[ExpertInfo | None], p2p: P2P) -> list[RemoteExpert | None]:
    experts: list[RemoteExpert | None] = []
    for info in infos:
        if info is not None:
            experts.append(RemoteExpert(info, p2p))
        else:
            experts.append(None)
    return experts


def create_remote_experts(
    infos: Sequence[ExpertInfo | None] | MPFuture, dht: DHT, return_future: bool = False
) -> list[RemoteExpert | None] | Future:
    if return_future:

        async def _unpack(infos_future: MPFuture, dht: DHT):
            p2p = await dht.replicate_p2p()
            return _create_remote_experts(await infos_future, p2p)

        return RemoteExpertWorker.run_coroutine(_unpack(infos, dht), return_future)

    p2p = RemoteExpertWorker.run_coroutine(dht.replicate_p2p())
    return _create_remote_experts(infos, p2p)


def batch_create_remote_experts(
    infos: Sequence[Sequence[ExpertInfo | None]] | MPFuture,
    dht: DHT,
    return_future: bool = False,
) -> list[list[RemoteExpert | None]] | Future:
    if return_future:

        async def _unpack(infos_future: MPFuture, dht: DHT):
            p2p = await dht.replicate_p2p()
            return [_create_remote_experts(i, p2p) for i in await infos_future]

        return RemoteExpertWorker.run_coroutine(_unpack(infos, dht), return_future)

    return [create_remote_experts(exps, dht) for exps in infos]


async def _backward_stream(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, metadata: bytes = b""
) -> list[torch.Tensor]:
    split = (part for tensor in serialized_tensors for part in split_for_streaming(tensor, DEFAULT_MAX_MSG_SIZE))
    meta = metadata or b""

    grad_inputs = await stub.rpc_backward_stream(
        amap_in_executor(
            lambda pair: runtime_pb2.ExpertRequest(
                uid=uid, tensors=[pair[1]], metadata=(meta if pair[0] == 0 else b"")
            ),
            iter_as_aiter(enumerate(split)),
        ),
    )
    tensors_stream = amap_in_executor(lambda msg: msg.tensors, grad_inputs)
    return await deserialize_tensor_stream(tensors_stream)


async def _backward_unary(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, metadata: bytes = b""
) -> list[torch.Tensor]:
    grad_inputs: runtime_pb2.ExpertResponse = await stub.rpc_backward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors), metadata=metadata or b"")
    )
    return [deserialize_torch_tensor(t) for t in grad_inputs.tensors]


async def expert_backward(
    uid: str,
    inputs_and_grads: Sequence[torch.Tensor],
    serialized_tensors: Iterable[runtime_pb2.Tensor],
    stub,
    metadata: bytes = b"",
) -> list[torch.Tensor]:
    size = 0
    for t in inputs_and_grads:
        size += t.element_size() * t.nelement()
        if size > MAX_UNARY_PAYLOAD_SIZE:
            return await _backward_stream(uid, serialized_tensors, stub, metadata=metadata)
    else:
        return await _backward_unary(uid, serialized_tensors, stub, metadata=metadata)


async def _forward_stream(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, metadata: bytes = b""
) -> list[torch.Tensor]:
    split = (p for t in serialized_tensors for p in split_for_streaming(t, DEFAULT_MAX_MSG_SIZE))
    meta = metadata or b""

    outputs = await stub.rpc_forward_stream(
        amap_in_executor(
            lambda pair: runtime_pb2.ExpertRequest(
                uid=uid, tensors=[pair[1]], metadata=(meta if pair[0] == 0 else b"")
            ),
            iter_as_aiter(enumerate(split)),
        ),
    )

    tensors_stream = amap_in_executor(lambda msg: msg.tensors, outputs)
    return await deserialize_tensor_stream(tensors_stream)


async def _forward_unary(
    uid: str, serialized_tensors: Iterable[runtime_pb2.Tensor], stub, metadata: bytes = b""
) -> list[torch.Tensor]:
    outputs: runtime_pb2.ExpertResponse = await stub.rpc_forward(
        runtime_pb2.ExpertRequest(uid=uid, tensors=list(serialized_tensors), metadata=metadata or b"")
    )
    return [deserialize_torch_tensor(t) for t in outputs.tensors]


async def expert_forward(
    uid: str,
    inputs: Sequence[torch.Tensor],
    serialized_tensors: Iterable[runtime_pb2.Tensor],
    stub,
    metadata: bytes = b"",
) -> list[torch.Tensor]:
    size = 0
    for t in inputs:
        size += t.element_size() * t.nelement()
        if size > MAX_UNARY_PAYLOAD_SIZE:
            return await _forward_stream(uid, serialized_tensors, stub, metadata=metadata)
    else:
        return await _forward_unary(uid, serialized_tensors, stub, metadata=metadata)


class _RemoteModuleCall(torch.autograd.Function):
    """Internal autograd-friendly call of a remote module. For applications, use RemoteExpert instead."""

    @staticmethod
    def forward(
        ctx,
        dummy: torch.Tensor,
        uid: str,
        stub: ConnectionHandlerStub,  # noqa: F821
        info: dict[str, Any],
        *inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Note: *inputs are flattened input tensors that follow the expert's info['input_schema']
        # detach to avoid pickling the computation graph
        inputs = tuple(tensor.cpu().detach() for tensor in inputs)
        ctx.uid, ctx.stub, ctx.info = uid, stub, info
        ctx.save_for_backward(*inputs)
        serialized_tensors = (
            serialize_torch_tensor(tensor, proto.compression)
            for tensor, proto in zip(inputs, nested_flatten(info["forward_schema"]))
        )
        deserialized_outputs = RemoteExpertWorker.run_coroutine(expert_forward(uid, inputs, serialized_tensors, stub))

        return tuple(deserialized_outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs) -> tuple[torch.Tensor | None, ...]:
        grad_outputs_cpu = tuple(tensor.cpu() for tensor in grad_outputs)
        inputs_and_grad_outputs = tuple(nested_flatten((ctx.saved_tensors, grad_outputs_cpu)))
        backward_schema = tuple(nested_flatten((ctx.info["forward_schema"], ctx.info["outputs_schema"])))
        serialized_tensors = (
            serialize_torch_tensor(tensor, proto.compression)
            for tensor, proto in zip(inputs_and_grad_outputs, backward_schema)
        )
        deserialized_grad_inputs = RemoteExpertWorker.run_coroutine(
            expert_backward(ctx.uid, inputs_and_grad_outputs, serialized_tensors, ctx.stub)
        )

        return (DUMMY, None, None, None, *deserialized_grad_inputs)
