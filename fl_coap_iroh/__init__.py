"""
fl_coap_iroh — Federated Learning with CoAP/CoRE control plane + Iroh/QUIC data plane.

Architecture:
  Control plane : CoAP / CoRE Resource Directory / CoRE Link Format
  Data plane    : Iroh (QUIC-based P2P with NAT traversal + relay fallback)

Topologies evaluated:
  A. Centralized FL — gRPC/HTTP baseline
  B. Centralized FL over Iroh  (isolates transport variable)
  C. Hierarchical edge FL (CoAP + gateway + Iroh)
  D. Hybrid relay-assisted P2P FL
  E. Decentralized / swarm FL  (exploratory)
"""
__version__ = "0.1.0"
