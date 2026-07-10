import sys
try:
    import torch
    print("torch", torch.__version__)
except Exception as e:
    print("torch FAIL:", e)
try:
    import iroh
    print("iroh OK")
except Exception as e:
    print("iroh FAIL:", e)
print("python", sys.version.split()[0])
