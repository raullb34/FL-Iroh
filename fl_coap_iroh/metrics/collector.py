"""
Metrics collection, aggregation, and CSV export.

Four metric categories (matching the experimental plan):
  A. Network   — transfer events: conn_type, latency, throughput, bytes
  B. FL        — round metrics: accuracy, loss, duration
  C. Infrastructure — round events: bytes to aggregator / p2p / relay
  D. CoAP      — discovery overhead: bytes, duration, n_nodes

All events are stored in memory and flushed to CSV on demand.
The CSV files are the primary output format for analysis notebooks.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Optional

from fl_coap_iroh.transport.iroh_node import TransferStats
from fl_coap_iroh.types import CoapEvent, ConnType, RoundEvent, RoundMetrics

log = logging.getLogger(__name__)


class MetricsCollector:
    """Thread-safe (GIL is sufficient for asyncio) metrics sink."""

    def __init__(
        self,
        node_id     : str,
        scenario    : str,
        architecture: str,
        output_dir  : str = "results",
    ) -> None:
        self.node_id      = node_id
        self.scenario     = scenario
        self.architecture = architecture
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        # Category A — network / transfer
        self._transfers: list[dict] = []
        # Category B — FL round metrics
        self._fl_metrics: list[dict] = []
        # Category C — infrastructure round events
        self._round_events: list[dict] = []
        # Category D — CoAP overhead
        self._coap_events: list[dict] = []

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    def record_transfer(self, stats: TransferStats, round_num: int, direction: str) -> None:
        """Category A: record one Iroh transfer event."""
        self._transfers.append({
            "node_id"           : self.node_id,
            "architecture"      : self.architecture,
            "scenario"          : self.scenario,
            "round"             : round_num,
            "direction"         : direction,
            "conn_type"         : stats.conn_type.value,
            "conn_time_ms"      : round(stats.conn_time_ms, 3),
            "bytes_payload"     : stats.bytes_payload,
            "bytes_on_wire"     : stats.bytes_on_wire,
            "duration_ms"       : round(stats.transfer_duration_ms, 3),
            "throughput_mbps"   : round(stats.throughput_mbps, 4),
            "timestamp"         : time.time(),
        })

    def record_round_metrics(self, m: RoundMetrics) -> None:
        """Category B: record client-side FL training metrics."""
        self._fl_metrics.append({
            "node_id"      : self.node_id,
            "architecture" : self.architecture,
            "scenario"     : self.scenario,
            **m.model_dump(),
        })

    def record_round_event(self, ev: RoundEvent) -> None:
        """Category C: record server-side round completion."""
        self._round_events.append({
            "architecture" : self.architecture,
            "scenario"     : self.scenario,
            **ev.model_dump(),
        })

    def record_coap_event(
        self,
        event_type : str,
        path       : str,
        bytes_total: int,
        duration_ms: float,
        n_nodes    : int = 1,
    ) -> None:
        """Category D: record a CoAP discovery or overhead measurement."""
        self._coap_events.append({
            "node_id"      : self.node_id,
            "architecture" : self.architecture,
            "scenario"     : self.scenario,
            "event_type"   : event_type,
            "path"         : path,
            "bytes_total"  : bytes_total,
            "duration_ms"  : round(duration_ms, 3),
            "n_nodes"      : n_nodes,
            "timestamp"    : time.time(),
        })

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, tag: Optional[str] = None) -> dict[str, Path]:
        """
        Write all accumulated metrics to CSV files.

        Returns:
            Dict mapping category name → file path.
        """
        prefix = tag or f"{self.architecture}_{self.scenario}"
        written: dict[str, Path] = {}

        pairs = [
            ("transfers",    self._transfers),
            ("fl_metrics",   self._fl_metrics),
            ("round_events", self._round_events),
            ("coap",         self._coap_events),
        ]
        for name, rows in pairs:
            if rows:
                path = self._dir / f"{prefix}_{name}.csv"
                _write_csv(path, rows)
                written[name] = path
                log.info("Exported %d rows → %s", len(rows), path)

        return written

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a compact summary dict for final logging."""
        if not self._round_events:
            return {"node_id": self.node_id, "architecture": self.architecture}

        rows = self._round_events
        accs  = [r["test_acc"]  for r in rows if r.get("test_acc")  is not None]
        durs  = [r["duration_sec"] for r in rows if r.get("duration_sec") is not None]
        n_ok  = sum(1 for r in rows if r.get("success"))

        total_t = len(self._transfers)
        n_direct = sum(1 for t in self._transfers if t["conn_type"] == ConnType.DIRECT.value)
        n_relay  = sum(1 for t in self._transfers if t["conn_type"] == ConnType.RELAY.value)

        return {
            "node_id"            : self.node_id,
            "architecture"       : self.architecture,
            "scenario"           : self.scenario,
            "rounds_total"       : len(rows),
            "rounds_success"     : n_ok,
            "test_acc_final"     : accs[-1] if accs else None,
            "test_acc_max"       : max(accs)  if accs else None,
            "duration_mean_sec"  : sum(durs) / len(durs) if durs else None,
            "pct_direct"         : 100 * n_direct / max(total_t, 1),
            "pct_relay"          : 100 * n_relay  / max(total_t, 1),
            "bytes_to_aggregator": sum(r.get("bytes_to_aggregator", 0) for r in rows),
        }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
