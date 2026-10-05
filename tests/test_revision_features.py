"""Resource Directory, CBOR negotiation, classification metrics, resource monitor,
and the Iroh control channel."""
from __future__ import annotations

import asyncio
import time

import pytest

from fl_coap_iroh.coap.resources import ResourceDirectory


def _entry(i, tags, energy=100.0):
    return {"node_id": f"n{i}", "tags": tags, "role": "client", "energy": energy,
            "iroh_endpoint": {"node_id_iroh": f"{i:064x}", "addrs": [], "relay_url": None,
                              "direct_capable": True}}


def test_rd_multi_tag_filter_and_paging():
    rd = ResourceDirectory()
    for i in range(10):
        rd.register(_entry(i, ["agri", f"r{i % 2}"] + (["soil"] if i < 3 else []), energy=10 * i))
    assert len(rd.lookup(["agri"], None, 0, 0, None)) == 10
    assert {e["node_id"] for e in rd.lookup(["agri", "r0", "soil"], None, 0, 0, None)} == {"n0", "n2"}
    assert len(rd.lookup(["agri"], None, 50, 0, None)) == 5          # energy >= 50
    assert [e["node_id"] for e in rd.lookup(["agri"], None, 0, 1, 3)] == ["n3", "n4", "n5"]


def test_rd_entries_expire():
    rd = ResourceDirectory()
    e = _entry(1, ["agri"])
    e["lt"] = -1
    rd.register(e)
    assert rd.lookup([], None, 0, 0, None) == []


async def test_rd_over_coap_with_cbor(unused_udp_port):
    from fl_coap_iroh.coap.client import FLCoapClient
    from fl_coap_iroh.coap.server import FLCoapServer
    from fl_coap_iroh.types import DatasetDescriptor, NodeCapabilities, NodeRole
    s = FLCoapServer("agg", NodeCapabilities(node_id="agg", role=NodeRole.AGGREGATOR),
                     DatasetDescriptor(dataset_id="x", dataset_name="x", samples=0, classes=[],
                                       iid=True, distribution="iid", feature_dim=[]),
                     coap_host="127.0.0.1", coap_port=unused_udp_port, enable_rd=True)
    await s.start()
    try:
        async with FLCoapClient("127.0.0.1", unused_udp_port) as c:
            for i in range(3):
                await c.rd_register(_entry(i, ["agri", "soil"] if i else ["air"]))
            hits, nbytes = await c.rd_lookup(["agri", "soil"], cbor=True)
            assert {h["node_id"] for h in hits} == {"n1", "n2"} and nbytes > 0
            json_hits, _ = await c.rd_lookup(["agri"], cbor=False)
            assert len(json_hits) == 2
            caps = await c.get_capabilities()   # Accept-less GET still works
            assert caps.node_id == "agg"
    finally:
        await s.stop()


async def test_rd_registration_respects_allowlist(unused_udp_port):
    from fl_coap_iroh.coap.client import FLCoapClient
    from fl_coap_iroh.coap.server import FLCoapServer
    from fl_coap_iroh.types import DatasetDescriptor, NodeCapabilities, NodeRole
    s = FLCoapServer("agg", NodeCapabilities(node_id="agg", role=NodeRole.AGGREGATOR),
                     DatasetDescriptor(dataset_id="x", dataset_name="x", samples=0, classes=[],
                                       iid=True, distribution="iid", feature_dim=[]),
                     coap_host="127.0.0.1", coap_port=unused_udp_port, enable_rd=True)
    s.set_registration_callback(lambda cid, ep: ep.node_id_iroh.endswith("1"))
    await s.start()
    try:
        async with FLCoapClient("127.0.0.1", unused_udp_port) as c:
            await c.rd_register(_entry(1, ["agri"]))
            with pytest.raises(RuntimeError, match="4.03"):
                await c.rd_register(_entry(2, ["agri"]))
        assert set(s.rd.entries) == {"n1"}
    finally:
        await s.stop()


def test_classification_report_imbalanced():
    from fl_coap_iroh.metrics.classification import classification_report, flat
    y_true = [0] * 6 + [1] * 2 + [2] * 2
    y_pred = [0] * 10                     # majority-class predictor
    r = classification_report(y_true, y_pred, 3)
    assert r["accuracy"] == pytest.approx(0.6)
    assert r["balanced_accuracy"] == pytest.approx(1 / 3)
    assert r["macro_f1"] < 0.3
    assert r["confusion_matrix"][1] == [2, 0, 0]
    assert "recall_Medio" in flat(r)


def test_resource_monitor_phases(monkeypatch):
    pytest.importorskip("psutil")
    monkeypatch.setenv("FL_POWER_SOURCE", "none")
    from fl_coap_iroh.metrics.resources import ResourceMonitor
    mon = ResourceMonitor(interval_s=0.05).start()
    with mon.phase("busy", 1):
        t = time.time()
        while time.time() - t < 0.4:
            sum(i * i for i in range(1000))
    mon.stop()
    (ph,) = mon.phases
    assert ph["phase"] == "busy" and ph["cpu_s"] > 0.1
    assert ph["rss_peak_mb"] > 1 and ph["energy_method"] == "model" and ph["energy_j"] > 0


async def test_control_channel_rejects_spoofed_nodeid(monkeypatch):
    """A peer cannot register another node's NodeId over fl-ctrl/1."""
    pytest.importorskip("iroh")
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("FL_MOCK_IROH", "0")
    monkeypatch.setenv("FL_CONN_CLASSIFY_SETTLE_S", "0.3")
    from fl_coap_iroh.fl.server import FLServer
    from fl_coap_iroh.transport.iroh_node import IrohTransportNode
    from fl_coap_iroh.types import NodeCapabilities, NodeRole, TrainingPolicy
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
        sk.bind(("127.0.0.1", 0)); port = sk.getsockname()[1]
    ds = torch.utils.data.TensorDataset(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))
    srv = FLServer("srv", torch.nn.Linear(2, 2), ds,
                   NodeCapabilities(node_id="srv", role=NodeRole.AGGREGATOR),
                   TrainingPolicy(min_clients=1), coap_port=port)
    srv_ep = await srv.start()
    honest, mallory = IrohTransportNode("honest"), IrohTransportNode("mallory")
    h_ep, m_ep = await honest.start(), await mallory.start()
    try:
        await mallory.send_control(srv_ep, {"type": "register", "client_id": "x",
                                            "iroh_endpoint": h_ep.model_dump()})
        await honest.send_control(srv_ep, {"type": "register", "client_id": "honest",
                                           "iroh_endpoint": h_ep.model_dump()})
        for _ in range(50):
            if "honest" in srv._clients:
                break
            await asyncio.sleep(0.1)
        assert set(srv._clients) == {"honest"}
        assert any("does not match" in r.reason for r in srv.allowlist.rejections)
    finally:
        await honest.stop(); await mallory.stop(); await srv.stop()


@pytest.fixture
def unused_udp_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
