"""Admission control (Ed25519 NodeId allow-list) and update binding."""
from __future__ import annotations

import asyncio
import os

import pytest

from fl_coap_iroh.transport.admission import AllowList, load_or_create_secret_key
from fl_coap_iroh.transport.iroh_node import (
    ALPN_FL_UPDATE,
    IrohTransportNode,
)


def test_secret_key_is_persistent(tmp_path):
    k1 = load_or_create_secret_key(tmp_path / "node.key")
    k2 = load_or_create_secret_key(tmp_path / "node.key")
    assert k1 == k2 and len(k1) == 32


def test_allowlist_file_parsing(tmp_path):
    f = tmp_path / "allow.txt"
    f.write_text("# members\nABCDEF  # farm A\n\n0123\n")
    al = AllowList.from_file(f)
    assert al.is_allowed("abcdef") and al.is_allowed("0123")
    assert not al.check("ffff", "test")
    assert len(al.rejections) == 1


def test_open_allowlist_admits_everyone():
    assert AllowList().check("anything", "test")


@pytest.fixture
def real_iroh(monkeypatch):
    pytest.importorskip("iroh")
    monkeypatch.setenv("FL_MOCK_IROH", "0")
    monkeypatch.setenv("FL_CONN_CLASSIFY_SETTLE_S", "0.5")


async def test_persistent_node_id(real_iroh, tmp_path):
    ids = []
    for _ in range(2):
        n = IrohTransportNode("x", secret_key_file=str(tmp_path / "x.key"))
        ep = await n.start()
        ids.append(ep.node_id_iroh)
        await n.stop()
    assert ids[0] == ids[1]


async def test_unauthorized_peer_is_rejected(real_iroh, tmp_path):
    good = IrohTransportNode("good", secret_key_file=str(tmp_path / "good.key"))
    bad  = IrohTransportNode("bad")
    good_ep = await good.start()
    await bad.start()

    server = IrohTransportNode("server", allowlist=AllowList({good_ep.node_id_iroh}, enforce=True))
    server_ep = await server.start()
    try:
        await good._send_bytes(server_ep, b"update-from-good", 1, ALPN_FL_UPDATE)
        payload, stats = await server._receive_bytes(ALPN_FL_UPDATE, timeout=20)
        assert payload == b"update-from-good"
        assert stats.peer_node_id == good_ep.node_id_iroh

        try:
            await asyncio.wait_for(
                bad._send_bytes(server_ep, b"poison", 1, ALPN_FL_UPDATE), timeout=20)
        except Exception:
            pass  # the connection may be closed before the write completes
        with pytest.raises(asyncio.TimeoutError):
            await server._receive_bytes(ALPN_FL_UPDATE, timeout=3)
        assert len(server.allowlist.rejections) == 1
        assert server.allowlist.rejections[0].channel == "iroh:fl-update/1"
    finally:
        for n in (good, bad, server):
            await n.stop()


async def test_update_from_unselected_peer_is_discarded(monkeypatch):
    """FLServer accepts one update per *selected* NodeId per round."""
    monkeypatch.setenv("FL_MOCK_IROH", "1")
    torch = pytest.importorskip("torch")
    from fl_coap_iroh.fl import server as server_mod
    from fl_coap_iroh.fl.server import FLServer
    from fl_coap_iroh.types import NodeCapabilities, NodeRole, TrainingPolicy

    monkeypatch.setattr(server_mod, "ROUND_TIMEOUT_SEC", 2.0)
    model = torch.nn.Linear(2, 2)
    ds = torch.utils.data.TensorDataset(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long))
    srv = FLServer("srv", model, ds, NodeCapabilities(node_id="srv", role=NodeRole.AGGREGATOR),
                   TrainingPolicy(min_clients=1), coap_port=_free_udp_port())
    srv_ep = await srv._transport.start()
    a = IrohTransportNode("a"); a_ep = await a.start()
    b = IrohTransportNode("b"); await b.start()
    params = {k: v.clone() for k, v in model.state_dict().items()}
    await b.send_tensors(srv_ep, params, 1, ALPN_FL_UPDATE)   # not selected
    await a.send_tensors(srv_ep, params, 1, ALPN_FL_UPDATE)   # selected
    await a.send_tensors(srv_ep, params, 1, ALPN_FL_UPDATE)   # duplicate
    updates, *_rest, discarded = await srv._collect_updates({a_ep.node_id_iroh: "a"}, 1)
    assert len(updates) == 1 and discarded == 1


def _free_udp_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_watchdog_restart_keeps_nodeid_and_queue(real_iroh):
    """A restarted endpoint keeps its NodeId (even without a key file) and
    still delivers to receivers waiting on the original accept queue."""
    rx, tx = IrohTransportNode("rx"), IrohTransportNode("tx")
    ep1 = await rx.start()
    await tx.start()
    try:
        pending = asyncio.create_task(rx._receive_bytes(ALPN_FL_UPDATE, timeout=40))
        ep2 = await rx.restart()
        assert ep2.node_id_iroh == ep1.node_id_iroh and rx.restarts == 1
        await tx._send_bytes(ep2, b"after-restart", 1, ALPN_FL_UPDATE)
        payload, _ = await pending
        assert payload == b"after-restart"
    finally:
        await rx.stop(); await tx.stop()


async def test_each_transfer_is_delivered_exactly_once(real_iroh):
    """Sender retries must not duplicate deliveries when the receiver closes
    the connection right after a verified receipt."""
    rx, tx = IrohTransportNode("rx"), IrohTransportNode("tx")
    rx_ep = await rx.start()
    await tx.start()
    try:
        for i in range(30):
            await tx._send_bytes(rx_ep, f"msg-{i}".encode(), i, ALPN_FL_UPDATE)
        got = []
        for _ in range(30):
            payload, _ = await rx._receive_bytes(ALPN_FL_UPDATE, timeout=20)
            got.append(payload)
        assert sorted(got) == sorted(f"msg-{i}".encode() for i in range(30))
        with pytest.raises(asyncio.TimeoutError):
            await rx._receive_bytes(ALPN_FL_UPDATE, timeout=3)
    finally:
        await rx.stop(); await tx.stop()
