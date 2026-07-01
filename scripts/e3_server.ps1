<#
E3 real-hardware SERVER launcher (Windows PC).

The PC runs the SERVER role because it uses the torch-free `_receive_bytes`
path (Windows torch is broken here) and iroh 0.35.0 works fine. It prints the
endpoint JSON that you paste into the RPi client, then waits for N connections
and writes the classified results (conn_type + active_addr) with the fixed
_active_wire_addr classifier.

Usage (from repo root):
    .\scripts\e3_server.ps1 net_nat1        # 30 transfers, scenario net_nat1
    .\scripts\e3_server.ps1 net_nat2 30
    .\scripts\e3_server.ps1 net_fw443 30

Outputs (results/e3/):
    server_endpoint.json                 - paste into the RPi client
    e3_nat_<scenario>_server.csv         - per-iter conn_type + active_addr
    e3_summary_<scenario>.csv            - pct_direct / pct_relay / pct_failed

IMPORTANT before running a WAN scenario, verify network isolation on the PC:
    Test-Connection 10.50.16.202 -Count 3   # RPi PRIVATE IP - MUST fail (100% loss)
If it succeeds you are co-located on the RPi LAN and the result is a confound.
#>
param(
    [Parameter(Mandatory = $true)] [string] $Scenario,
    [int] $NIter = 30
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

$py = 'C:\Users\Raul\AppData\Local\Programs\Python\Python311\python.exe'
if (-not (Test-Path $py)) { throw "Python not found at $py" }

$env:FL_CONN_DEBUG_ADDRS = '1'   # dump full addrs list for the audit trail
Remove-Item Env:\FL_MOCK_IROH -ErrorAction SilentlyContinue

Write-Host "== E3 SERVER (PC) scenario=$Scenario n_iter=$NIter ==" -ForegroundColor Cyan
Write-Host "Reminder: confirm 'Test-Connection <RPi-private-IP>' FAILS before trusting WAN results." -ForegroundColor Yellow

& $py -u -m experiments.e3_nat_traversal --role server --n-iter $NIter --scenario $Scenario
