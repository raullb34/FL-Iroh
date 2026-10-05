"""
Federation admission control based on Iroh node identities (Ed25519).

Every Iroh node is identified by its Ed25519 public key (the NodeId), and the
QUIC/TLS 1.3 handshake proves possession of the matching secret key.  A peer
therefore cannot claim a NodeId it does not own, so an allow-list of NodeIds
is sufficient to authorise federation members without a PKI or a VPN
coordination server.

This module provides:
  * ``load_or_create_secret_key``: a persistent 32-byte secret key so that a
    node keeps the same NodeId across restarts (required for an allow-list).
  * ``AllowList``: a set of authorised NodeIds loaded from a text file
    (one hex NodeId per line, ``#`` comments allowed), plus a log of rejected
    connection / registration attempts for auditing and experiments (E9).

Environment variables (used when the corresponding constructor argument is
not given):
  FL_SECRET_KEY_FILE   path of the persistent secret key file
  FL_ALLOWLIST_FILE    path of the allow-list file (unset = open federation)
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SECRET_KEY_BYTES = 32


def load_or_create_secret_key(path: str | os.PathLike) -> bytes:
    """Return the 32-byte secret key stored at *path*, creating it if absent."""
    p = Path(path)
    if p.exists():
        key = p.read_bytes()
        if len(key) != SECRET_KEY_BYTES:
            raise ValueError(f"{p}: expected {SECRET_KEY_BYTES} bytes, got {len(key)}")
        return key
    p.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(SECRET_KEY_BYTES)
    # Create with owner-only permissions where the OS supports it.
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key)
    log.info("Generated new node secret key → %s", p)
    return key


def _normalise(node_id: str) -> str:
    return node_id.strip().lower()


@dataclass
class Rejection:
    timestamp : float
    node_id   : str
    channel   : str      # "iroh:<alpn>" or "coap:/fl/register"
    reason    : str


@dataclass
class AllowList:
    """Set of NodeIds authorised to take part in the federation.

    ``enforce=False`` (the default when no allow-list file is configured)
    admits every peer, preserving the behaviour of the original prototype.
    """
    node_ids  : set[str] = field(default_factory=set)
    enforce   : bool     = False
    rejections: list[Rejection] = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "AllowList":
        ids: set[str] = set()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                ids.add(_normalise(line))
        log.info("Loaded allow-list with %d NodeId(s) from %s", len(ids), path)
        return cls(node_ids=ids, enforce=True)

    @classmethod
    def from_env(cls) -> "AllowList":
        path = os.environ.get("FL_ALLOWLIST_FILE")
        if path:
            return cls.from_file(path)
        return cls()

    # ------------------------------------------------------------------
    def add(self, node_id: str) -> None:
        self.node_ids.add(_normalise(node_id))

    def is_allowed(self, node_id: Optional[str]) -> bool:
        if not self.enforce:
            return True
        return node_id is not None and _normalise(node_id) in self.node_ids

    def check(self, node_id: Optional[str], channel: str) -> bool:
        """Return whether *node_id* is admitted, recording a rejection if not."""
        if self.is_allowed(node_id):
            return True
        self.rejections.append(Rejection(
            timestamp = time.time(),
            node_id   = node_id or "",
            channel   = channel,
            reason    = "node_id not in allow-list",
        ))
        log.warning("Rejected unauthorised peer %s on %s", (node_id or "?")[:16], channel)
        return False
