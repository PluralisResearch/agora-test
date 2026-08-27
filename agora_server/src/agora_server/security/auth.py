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

import os
import sys

from typing import TypeVar

from agora_server.hivemind import PeerID
from agora_server.hivemind.proto import crypto_pb2
from agora_server.hivemind.utils.crypto import RSAPrivateKey
from agora_server.hivemind.utils.logging import get_logger
from agora_server.security.authorizer import PluralisAuthorizer
from agora_server.utils.node_info import NodeInfo


AuthorizerType = TypeVar("AuthorizerType", bound=PluralisAuthorizer)

logger = get_logger(__name__)


def save_identity(private_key: RSAPrivateKey, identity_path: str) -> None:
    """Save private key to file.

    Args:
        private_key (RSAPrivateKey): Local private key.
        identity_path (str): Path to save the key.

    Raises:
        FileNotFoundError: Can't create file.
    """
    protobuf = crypto_pb2.PrivateKey(key_type=crypto_pb2.KeyType.RSA, data=private_key.to_bytes())

    try:
        with open(identity_path, "wb") as f:
            f.write(protobuf.SerializeToString())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The directory `{os.path.dirname(identity_path)}` for saving the identity does not exist"
        ) from None
    os.chmod(identity_path, 0o400)


def load_identity_key(identity_path: str) -> RSAPrivateKey:
    """Load the persisted per-peer RSA identity key into a named instance,
    generating and persisting a fresh one if the file does not exist yet.

    This is the peer's single identity: p2pd derives ``peer_id`` from it, the
    auth_server binds AccessTokens to it, and every Metadata Store record
    envelope is signed with it. Callers thread the returned instance explicitly
    (RedisDHT, authorizer); nothing here touches the process-wide singleton.

    Args:
        identity_path (str): Path to load/save the key.

    Returns:
        RSAPrivateKey: The persisted identity key.
    """
    if os.path.exists(identity_path):
        with open(identity_path, "rb") as f:
            key_data = crypto_pb2.PrivateKey.FromString(f.read()).data
        return RSAPrivateKey.from_der_bytes(key_data)

    private_key = RSAPrivateKey()
    save_identity(private_key, identity_path)
    return private_key


def authorize_with_pluralis(
    user_token: str,
    role: str,
    auth_server: str,
    identity_path: str,
    node_info: NodeInfo | None = None,
    authorizer_cls: type[AuthorizerType] = PluralisAuthorizer,
    expert_uid: str | None = None,
    **kwargs,
) -> AuthorizerType:
    """Generate local keys and send authorization request to join the run.

    Args:
        user_token (str): Authentication token.
        role (str): Role in the swarm.
        auth_server (str): Authorization server URL.
        identity_path (str): Path to save/load private key.
        node_info (NodeInfo | None): Information about the node. Defaults to None.
        authorizer_cls (type[AuthorizerType]): Class to instantiate the authorizer. Defaults to PluralisAuthorizer.
        expert_uid (str | None): Expert UID (e.g. head.0.42). Defaults to None.

    Returns:
        AuthorizerType: Authorizer instance.
    """
    logger.info("Authorization started...")

    # Load (or generate + persist) the identity key as a named instance. It is
    # threaded explicitly into the authorizer below (and by callers into
    # RedisDHT); set_process_wide keeps the legacy DHT-mode components that
    # still default to the process-wide key (record validators, owner tags,
    # kill-switch) on the same persisted identity.
    private_key = load_identity_key(identity_path)
    RSAPrivateKey.set_process_wide(private_key)

    # Get static peer id
    with open(identity_path, "rb") as f:
        peer_id = str(PeerID.from_identity(f.read()))

    # Authorize
    try:
        authorizer = authorizer_cls(
            peer_id=peer_id,
            user_token=user_token,
            role=role,
            auth_server=auth_server,
            node_info=node_info,
            expert_uid=expert_uid,
            local_private_key=private_key,
            **kwargs,
        )

        authorizer.join_experiment()
        return authorizer
    except Exception as e:
        logger.error(f"Authorization failed: {e}. Exiting run.")
        sys.exit(1)


def apply_shard_assignment(args: dict, authorizer: PluralisAuthorizer) -> None:
    """Overwrite launch args with the authorizer's data-shard assignment, when present.

    The assignment carries the corpus location (the URL is never committed to
    configs) plus a fresh shard index. Server values win over configured ones.
    The configured registry corpus name stays: it pins the manifest identity
    and reader overrides the assigned public URL must resolve to.

    Args:
        args (dict): Launch args about to be passed to ``Server.create``; mutated in place.
        authorizer (PluralisAuthorizer): Authorizer holding the parsed response fields.
    """
    if authorizer.pithos_shard_index is None:
        return
    assigned = {
        "pithos_shard_index": authorizer.pithos_shard_index,
        "pithos_shard_count": authorizer.pithos_shard_count,
        "pithos_shards_per_head": authorizer.pithos_shards_per_head,
        "pithos_corpus_uri": authorizer.pithos_corpus_uri,
    }
    conflicting = {k: args[k] for k in assigned if args.get(k) is not None and args[k] != assigned[k]}
    if conflicting:
        logger.warning(f"Authorizer shard assignment overrides configured values {conflicting}")
    args.update(assigned)
    logger.info(f"Authorizer assigned data shard {assigned['pithos_shard_index']}/{assigned['pithos_shard_count']}")
