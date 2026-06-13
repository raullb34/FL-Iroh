import csv
from pathlib import Path

# Check province names in the CSV
csv_path = Path("data/air-quailty/datasets/calidad-del-aire-datos-historicos-diarios-cyl.csv")
provs = set()
estaciones = set()
with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=";")
    for row in reader:
        prov = row.get("Provincia", "").strip()
        est  = row.get("Estación", "").strip()
        if prov:
            provs.add(prov)
        # check for Ponferrada-type stations
        if "Ponferrada" in est or "Compostilla" in est or "Cortiguera" in est:
            estaciones.add((prov, est))

print("Provinces in CSV:")
for p in sorted(provs):
    print(f"  repr={repr(p)}")

print("\nPonferrada-related stations (prov, estacion):")
for x in sorted(estaciones):
    print(f"  {x}")

# Check NO2 distribution to understand ICA thresholds
import statistics
no2_vals = []
with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter=";")
    for row in reader:
        v = row.get("NO2 (ug/m3)", "").strip()
        if v and v not in ("", "None", "-"):
            try:
                no2_vals.append(float(v))
            except ValueError:
                pass

no2_vals.sort()
n = len(no2_vals)
print(f"\nNO2 distribution (n={n}):")
print(f"  min={no2_vals[0]:.1f}  max={no2_vals[-1]:.1f}  mean={statistics.mean(no2_vals):.1f}  median={statistics.median(no2_vals):.1f}")
pct = [10, 25, 50, 66, 75, 90, 95, 99]
for p in pct:
    idx = int(n * p / 100)
    print(f"  p{p:02d}={no2_vals[idx]:.1f}")
