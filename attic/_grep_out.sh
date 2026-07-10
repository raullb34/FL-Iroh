#!/usr/bin/env bash
F="$1"
grep -aE "^(server addrs|t=)" "$F"
