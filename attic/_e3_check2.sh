#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/bin/python
echo "FL_MOCK_IROH=[$FL_MOCK_IROH]"
echo "--- transport import (server path, torch-free) ---"
$PY -u -c 'from fl_coap_iroh.transport.iroh_node import IrohTransportNode; print("transport import OK (no torch needed)")' 2>&1 | head -20
echo "EXIT=${PIPESTATUS[0]}"
