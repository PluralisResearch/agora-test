from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import msgpack
import pytest

from agora import run_server
from agora_server.core import envelope
from agora_server.core.metadata_store_client import MetadataStoreError
from agora_server.core.redis_dht import RedisDHT
from agora_server.core.server.dht_handler import format_expert_dht_value
from agora_server.hivemind.p2p import PeerID
from agora_server.hivemind.utils import get_dht_time
from agora_server.hivemind.utils.crypto import RSAPrivateKey
from agora_server.types import GhostPhase, ServerCreationError


MESSAGE = (
    "Peer_id already present in the system. Make sure no other instances of Agora "
    "are running from this host and try again in few minutes."
)
STAGES = ["head", *[f"body{i}" for i in range(1, 12)], "tail"]
OTHER_PEER = PeerID.from_base58("QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt")


@pytest.fixture
def store(monkeypatch):
    private_key = RSAPrivateKey()
    peer = PeerID.from_rsa_pubkey(private_key.get_public_key().to_bytes())
    state = SimpleNamespace(
        args={"metadata_store_url": "https://metadata.test", "num_stages": 13},
        authorizer=Mock(peer_id=peer.to_base58(), local_private_key=private_key),
        peer=peer,
        records={},
        requests=[],
        failure_key=None,
        failure=None,
        metadata=None,
    )

    def respond(request):
        assert request.url.path == "/v1/kv:get", "Startup must only read metadata"
        assert request.headers["authorization"] == "Bearer test-token"
        key = msgpack.loads(request.content, raw=False)["key"]
        state.requests.append(key)
        if key == state.failure_key:
            if isinstance(state.failure, Exception):
                raise state.failure
            return httpx.Response(state.failure)
        if key not in state.records:
            return httpx.Response(404)
        return httpx.Response(200, content=msgpack.dumps(state.records[key], use_bin_type=True))

    with httpx.Client(base_url=state.args["metadata_store_url"], transport=httpx.MockTransport(respond)) as client:

        def create_metadata(**kwargs):
            state.metadata = RedisDHT(http_client=client, **kwargs)
            return state.metadata

        monkeypatch.setattr(run_server, "RedisDHT", create_metadata)
        monkeypatch.setattr(run_server, "MetadataStoreTokenProvider", lambda authorizer: lambda: "test-token")
        yield state
        if state.metadata is not None:
            assert state.metadata._p2p is None
            assert state.metadata._client._closing, "Startup reads must release the metadata client"


def add_record(store, key, value, subkey=None):
    encoded = envelope.build(
        value, get_dht_time() + 60, key=key, subkey=subkey, private_key=store.authorizer.local_private_key
    )
    if subkey is None:
        store.records[key] = {"kind": "single", "envelope": encoded}
    else:
        record = store.records.setdefault(key, {"kind": "dict", "entries": []})
        record["entries"].append({"subkey": subkey, "envelope": encoded})


@pytest.mark.parametrize("addresses", [[], ["/ip4/127.0.0.1/tcp/1234"]])
def test_address_book_duplicate(store, addresses):
    key = f"peer:{store.peer}"
    add_record(store, key, addresses)
    with pytest.raises(ServerCreationError) as error:
        run_server._check_peer_id_available(store.args, store.authorizer)
    assert str(error.value) == MESSAGE
    assert store.requests == [key]


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("phase", list(GhostPhase))
def test_expert_duplicate_without_address_book(store, stage, phase):
    # The duplicate can have any UID and coexist with other peers in any stage.
    key = f"{stage}.0."
    add_record(store, key, [f"{stage}.0.1", str(OTHER_PEER)], subkey=b"other")
    value = format_expert_dht_value(store.peer, phase, 42)
    add_record(store, key, [f"{stage}.0.99", value], subkey=b"existing")
    with pytest.raises(ServerCreationError) as error:
        run_server._check_peer_id_available(store.args, store.authorizer)
    assert str(error.value) == MESSAGE


def test_unused_peer_with_other_experts(store):
    for stage in STAGES:
        for phase in GhostPhase:
            add_record(
                store,
                f"{stage}.0.",
                [f"{stage}.0.{phase.value}", format_expert_dht_value(OTHER_PEER, phase, 42)],
                subkey=str(phase.value).encode(),
            )
    run_server._check_peer_id_available(store.args, store.authorizer)
    assert store.requests == [f"peer:{store.peer}", *[f"{stage}.0." for stage in STAGES]]


def test_missing_or_expired_records_allow_startup(store):
    # The Metadata Store returns 404 for absent or expired records.
    run_server._check_peer_id_available(store.args, store.authorizer)
    assert len(store.requests) == 14


@pytest.mark.parametrize("key", ["address", "body11.0."])
@pytest.mark.parametrize("failure", [401, 503, httpx.ReadTimeout("store unavailable")])
def test_failed_read_aborts_startup(store, key, failure):
    store.failure_key = f"peer:{store.peer}" if key == "address" else key
    store.failure = failure
    with pytest.raises((MetadataStoreError, httpx.ReadTimeout)):
        run_server._check_peer_id_available(store.args, store.authorizer)


def test_no_metadata_store_skips_check(store):
    run_server._check_peer_id_available({}, store.authorizer)
    assert store.metadata is None


@pytest.mark.parametrize("download", [False, True])
def test_main_exits_before_server_creation_on_duplicate(store, monkeypatch, download):
    add_record(store, f"peer:{store.peer}", [])
    args = dict(
        store.args,
        token="test-token",
        email="",
        active_period_timeout=60,
        initial_peers=[],
        max_batch_size=1,
        auth_server="https://auth.test",
        identity_path="unused.key",
        announce_maddrs="/ip4/127.0.0.1/tcp/1234",
        host_maddrs="/ip4/0.0.0.0/tcp/1234",
        load_state_from_peer=not download,
    )
    monkeypatch.setattr(run_server, "parse_args", lambda: args)
    monkeypatch.setattr(run_server, "PluralisLogger", Mock(return_value=Mock(root_logger=Mock(handlers=[]))))
    monkeypatch.setattr(run_server, "LogMonitor", Mock())
    monkeypatch.setattr(run_server, "clean_tmp", Mock())
    monkeypatch.setattr(
        run_server,
        "get_node_info",
        lambda *a, **kw: SimpleNamespace(
            latency=1,
            device_name="cuda",
            gpu_memory=80,
        ),
    )
    monkeypatch.setattr(run_server, "authorize_with_pluralis", lambda **kw: store.authorizer)
    downloader = Mock()
    if download:
        downloader.ensure_downloaded.side_effect = RuntimeError("Download cancelled")
    monkeypatch.setattr(run_server, "StateDownloader", Mock(return_value=Mock(start=lambda: downloader)))
    create = Mock()
    monkeypatch.setattr(run_server.Server, "create", create)
    logger = Mock()
    monkeypatch.setattr(run_server, "logger", logger)

    with pytest.raises(SystemExit) as error:
        run_server.main()
    assert error.value.code == 1
    create.assert_not_called()
    logger.error.assert_called_with(f"Server failed during startup: {MESSAGE}")
    store.authorizer.begin_shutdown.assert_called_once()
    store.authorizer.cancel_queue_wait.assert_called_once()
    if download:
        downloader.cancel.assert_called()
