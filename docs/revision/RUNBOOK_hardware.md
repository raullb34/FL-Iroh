# Runbook: campañas con hardware para la revisión (R1.1, R1.2, R1.4, R1.5, R1.7, R1.3)

Todo se ejecuta desde la raíz del repo, en la rama `revision-r1`.
Sirve para 1 RPi + PC; con más RPi se repite el lado cliente en cada una.

## 0. Preparación (una sola vez por equipo)

```bash
# RPi / Linux / coautores
scripts/join_federation.sh --keygen rpi01      # crea .venv-edge y keys/rpi01.key, imprime el NodeId
# PC coordinador
.venv/bin/pip install -e ".[edge,baselines]"
.venv/bin/fl-keygen keys/server.key
```

Todos los NodeId de clientes van en `allow.txt` (uno por línea) en el coordinador.

**Etiqueta cada red** antes de medir en ella. Sirve para clasificar el NAT según RFC 4787 (R1.1):
```bash
.venv-edge/bin/python scripts/nat_classify.py --label rpi01-fibra-ISP2-ciudadX --out results/nat
```
El resultado va a `results/nat/`. Repítelo en cada red: fibra, cada operador móvil y cada casa de coautor.

## 1. Relay forzado, recuperación y throughput por relay (R1.2)

Coordinador (servidor, identidad persistente):
```bash
.venv/bin/python -m experiments.e12_connectivity --role server --key keys/server.key \
    --duration 3000 --endpoint-out results/e12/server_ep.json --label campaign1
```
Copia `results/e12/server_ep.json` a la RPi y ejecuta como root, porque hace falta nftables:
```bash
sudo .venv-edge/bin/python -m experiments.e12_connectivity --role client \
  --server-endpoint results/e12/server_ep.json --label rpi01-relay --key keys/rpi01.key \
  --duration 2400 --interval 5 --payload 1000000 \
  --events "300:block_udp,1200:unblock_udp,1500:block_udp,2100:unblock_udp"
```
Salida en `results/e12/e12_rpi01-relay_client_{transfers,events,summary}`:
- `s_to_first_relay` tras `block_udp`: latencia de fallback al relay.
- `s_to_first_direct` tras `unblock_udp`: tiempo de vuelta a la ruta directa.
- `udp_blocked.goodput_mbps_median`: throughput por relay.

## 2. Desconexión real y cambio de IP/NAT (R1.7)

Con el mismo servidor en marcha:
```bash
# (a) corte automático del enlace: 60 s sin wlan0 al minuto 5, 120 s al minuto 15
sudo .venv-edge/bin/python -m experiments.e12_connectivity --role client \
  --server-endpoint results/e12/server_ep.json --label rpi01-outage --key keys/rpi01.key \
  --duration 1500 --interval 5 --events "300:link_down:wlan0:60,900:link_down:wlan0:120"

# (b) cambio manual Wi-Fi → hotspot 4G (IP y mapping NAT cambian)
sudo .venv-edge/bin/python -m experiments.e12_connectivity --role client \
  --server-endpoint results/e12/server_ep.json --label rpi01-wifi2lte --key keys/rpi01.key \
  --duration 1200 --interval 5
#   justo antes de cambiar de red:   sudo pkill -USR1 -f e12_connectivity   (marca el evento)
```
En `*_events.csv` aparecen `s_to_first_ok` (reconexión), `failed_until_first_ok` y `local_ip_changed`.

**Reincorporación a rondas FL**: repite la desconexión durante una federación real (sección 4) y mira `clients_participated` por ronda en `results/wan/server/*round_events.csv`.

## 3. Repetición de E3 en frío con el clasificador corregido (R1.1)

El clasificador antiguo contaba `MIXED` como directo (ver `scripts/e3_reclassify.py`). Hay que repetir la campaña en frío en cada red nueva:
```bash
scripts/_e3_server.sh <escenario>                        # PC
scripts/e3_repeat_establish.sh <escenario> 30 @results/e3/server_endpoint.json   # RPi
python scripts/e3_reclassify.py                          # tabla direct / mixed / relay / failed
```
Nombra el escenario por la red real (p. ej. `net_movistar_fibra`, `net_vodafone_4g`) y guarda al lado el JSON de `nat_classify`.

## 4. Federación FL completa en varias RPi sobre WAN (R1.5) + recursos (R1.4)

Coordinador (puede estar detrás de NAT):
```bash
scripts/run_wan_server.sh allow.txt 2 30          # espera ≥2 clientes, 30 rondas
# comparte results/wan/server/server_endpoint.json con los clientes
```
Cada RPi o coautor, con una provincia distinta (índices 0–8):
```bash
scripts/join_federation.sh server_endpoint.json rpi01 0
```
Cada cliente escribe `results/wan/<id>/*_resource_phases.csv` con CPU-s, RSS pico y energía por fase (`recv`, `train`, `send`) y ronda.

**Energía** (la columna `energy_method` indica cómo se obtuvo):
- `hwmon`: medida real si hay un INA219/INA226 con `dtoverlay=i2c-sensor,ina219` en `/boot/firmware/config.txt`. Cuesta unos 5 €.
- `pmic`: medida real **en Raspberry Pi 5** (`vcgencmd pmic_read_adc`), sin hardware extra. Si vais a comprar RPi, las RPi 5 resuelven el medidor.
- `model`: estimación P = P_idle + (P_max − P_idle)·u, con valores de RPi 4B. Declarar como estimación en el paper.
- Con un medidor USB con registro (FNIRSI FNB58 o similar), su CSV se alinea después por timestamps de pared (`t_wall` en `*_resource_samples.csv`).

## 5. Flower + Tailscale en vivo, mismo hardware (R1.3)

Tailscale instalado y autenticado en PC y RPi (`tailscale ip -4`).
```bash
# PC
.venv/bin/python -m experiments.e8_flower_tailscale --mode server --server-address 0.0.0.0:8080 \
   --dataset crop --n-clients 2 --rounds 30 --results-dir results/e8_live
# RPi (y un segundo cliente en otro equipo)
FL_RESOURCE_MONITOR=1 .venv-edge/bin/python -m experiments.e8_flower_tailscale --mode client \
   --server-address <IP-tailscale-PC>:8080 --dataset crop --n-clients 2 --client-id 0 \
   --results-dir results/e8_live
```
El monitor incluye el proceso `tailscaled`, así que se compara CPU/RSS/energía del cliente más el daemon frente a FL-Iroh (sección 4 con `--dataset crop`). Apunta también los pasos y el tiempo de puesta en marcha de cada stack; es el eje operativo de la Tabla 8.

## 6. Experimentos sin hardware (ya automatizados)

| Experimento | Comando | Estado |
|---|---|---|
| E9 admisión Ed25519 | `python -m experiments.e9_admission --role local --n 50` | hecho (loopback); repetir PC↔RPi con `--role server/client` |
| E6b escalado CoAP/RD | `python -m experiments.e6b_coap_scaling` | hecho, hasta 5000 nodos |
| E11 red degradada | `sudo scripts/run_netem_sweep.sh 20` | en marcha / repetir con el código congelado |
| E7 métricas por clase | `python -m experiments.e7_air_quality_fl --all-seeds` | lanzar en el HPC **sin** `FL_MOCK_IROH=1` |
