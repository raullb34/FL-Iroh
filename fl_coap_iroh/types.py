"""
Shared data types and Pydantic schemas for fl_coap_iroh.

These are the canonical data structures exchanged:
  - via CoAP (control plane descriptors, JSON/CBOR)
  - via Iroh (data plane metadata embedded in framing headers)
  - in metrics CSV exports
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Node roles and status
# ---------------------------------------------------------------------------

class NodeRole(str, Enum):
    CLIENT      = "client"
    GATEWAY     = "gateway"
    AGGREGATOR  = "aggregator"
    RELAY       = "relay"


class EnergySource(str, Enum):
    BATTERY = "battery"
    AC      = "ac"
    SOLAR   = "solar"
    UNKNOWN = "unknown"


class NodeStatus(str, Enum):
    READY        = "ready"
    TRAINING     = "training"
    AGGREGATING  = "aggregating"
    IDLE         = "idle"
    UNAVAILABLE  = "unavailable"


# ---------------------------------------------------------------------------
# CoAP resource descriptors  (published at /fl/* and /iroh/endpoint)
# ---------------------------------------------------------------------------

class ComputeCapabilities(BaseModel):
    cpu_cores : int   = 1
    ram_mb    : int   = 512
    gpu       : bool  = False
    tpu       : bool  = False
    arch      : str   = "x86_64"


class EnergyState(BaseModel):
    source    : EnergySource = EnergySource.UNKNOWN
    level_pct : float        = Field(default=100.0, ge=0.0, le=100.0)
    plugged   : bool         = True


class AvailabilityInfo(BaseModel):
    status               : NodeStatus    = NodeStatus.IDLE
    max_rounds_per_day   : int           = 24
    preferred_window_utc : Optional[str] = None  # e.g. "02:00-06:00"


class NodeCapabilities(BaseModel):
    """Published at /fl/capabilities"""
    node_id       : str
    role          : NodeRole
    fl_frameworks : list[str]          = Field(default_factory=lambda: ["custom"])
    compute       : ComputeCapabilities = Field(default_factory=ComputeCapabilities)
    energy        : EnergyState         = Field(default_factory=EnergyState)
    availability  : AvailabilityInfo    = Field(default_factory=AvailabilityInfo)
    version       : str                = "0.1.0"
    timestamp     : float              = Field(default_factory=time.time)


class DatasetDescriptor(BaseModel):
    """Published at /fl/dataset — metadata only, never raw data"""
    dataset_id   : str
    dataset_name : str            # "cifar10" | "mnist" | "har"
    samples      : int
    classes      : list[int]
    iid          : bool
    distribution : str            # "iid" | "dirichlet-alpha-0.5"
    feature_dim  : list[int]
    format       : str = "pytorch-tensor"


class ModelDescriptor(BaseModel):
    """Published at /fl/model"""
    model_id    : str
    model_name  : str
    round       : int
    params_count: int
    size_bytes  : int
    sha256      : str
    timestamp   : float = Field(default_factory=time.time)


class UpdateDescriptor(BaseModel):
    """Published at /fl/update"""
    node_id        : str
    round          : int
    samples_used   : int
    iroh_blob_hash : Optional[str] = None  # future: Iroh blob hash
    status         : str           = "pending"  # pending|sent|received|aggregated
    timestamp      : float         = Field(default_factory=time.time)


class RoundMetrics(BaseModel):
    """Published at /fl/metrics"""
    node_id      : str
    round        : int
    train_loss   : float
    train_acc    : float
    val_loss     : Optional[float] = None
    val_acc      : Optional[float] = None
    samples_used : int
    duration_sec : float
    timestamp    : float = Field(default_factory=time.time)


class TrainingPolicy(BaseModel):
    """Published at /fl/policy"""
    batch_size    : int   = 32
    learning_rate : float = 0.01
    local_epochs  : int   = 1
    max_rounds    : int   = 100
    min_clients   : int   = 2
    fraction_fit  : float = 1.0
    proximal_mu   : float = 0.0   # FedProx; 0.0 == FedAvg


class RoundState(BaseModel):
    """Published at /fl/round"""
    round                : int
    status               : str            # waiting|training|aggregating|done|failed
    participants_expected: int
    participants_joined  : list[str]      = Field(default_factory=list)
    participants_done    : list[str]      = Field(default_factory=list)
    start_time           : Optional[float] = None
    end_time             : Optional[float] = None


class IrohEndpoint(BaseModel):
    """Published at /iroh/endpoint"""
    node_id_iroh   : str
    addrs          : list[str]
    relay_url      : Optional[str] = None
    direct_capable : bool          = True
    last_seen      : float         = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Network & metrics events  (for CSV export)
# ---------------------------------------------------------------------------

class ConnType(str, Enum):
    DIRECT  = "direct"
    RELAY   = "relay"
    UNKNOWN = "unknown"


class ConnectionEvent(BaseModel):
    peer_node_id : str
    conn_type    : ConnType
    conn_time_ms : float
    timestamp    : float = Field(default_factory=time.time)


class TransferEvent(BaseModel):
    peer_node_id       : str
    direction          : str       # "send" | "recv"
    bytes_transferred  : int
    duration_ms        : float
    throughput_mbps    : float
    conn_type          : ConnType
    round              : int
    timestamp          : float = Field(default_factory=time.time)


class RoundEvent(BaseModel):
    """One row per completed/failed FL round (server perspective)."""
    round                : int
    architecture         : str
    scenario             : str
    n_clients            : int
    clients_participated : int
    duration_sec         : float
    success              : bool
    test_acc             : Optional[float] = None
    test_loss            : Optional[float] = None
    bytes_to_aggregator  : int             = 0
    bytes_p2p_direct     : int             = 0
    bytes_relay          : int             = 0
    timestamp            : float           = Field(default_factory=time.time)


class CoapEvent(BaseModel):
    """One row per CoAP discovery/overhead measurement."""
    node_id      : str
    scenario     : str
    architecture : str
    event_type   : str    # "discovery" | "get_caps" | "get_endpoint" | "post_metrics"
    path         : str
    bytes_total  : int
    duration_ms  : float
    n_nodes      : int    = 1
    timestamp    : float  = Field(default_factory=time.time)
