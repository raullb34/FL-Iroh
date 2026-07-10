#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/bin/python
echo "FL_MOCK_IROH=[$FL_MOCK_IROH]"
echo "--- iroh ---"
$PY -c 'import iroh; print("iroh OK", getattr(iroh, "__version__", "?"))' 2>&1 | head -5
echo "--- torch (lazy, not needed by server) ---"
$PY -c 'import torch; print("torch OK")' 2>&1 | head -3
echo "--- import transport node (server path, should NOT need torch) ---"
$PY -c 'from fl_coap_iroh.transport.iroh_node import IrohTransportNode; print("transport import OK")' 2>&1 | head -5
