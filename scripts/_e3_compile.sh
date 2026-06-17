#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
./.venv/bin/python -m py_compile fl_coap_iroh/transport/iroh_node.py experiments/e3_nat_traversal.py && echo COMPILE_OK
