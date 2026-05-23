import sys
print(f"Python: {sys.version}")
deps = {}
for name, mod in [
    ("torch",    "torch"),
    ("flwr",     "flwr"),
    ("opacus",   "opacus"),
    ("fastapi",  "fastapi"),
    ("numpy",    "numpy"),
    ("pandas",   "pandas"),
    ("sklearn",  "sklearn"),
    ("scipy",    "scipy"),
    ("httpx",    "httpx"),
    ("uvicorn",  "uvicorn"),
    ("cryptography", "cryptography"),
    ("redis",    "redis"),
]:
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "ok")
        deps[name] = v
        print(f"  [OK] {name}=={v}")
    except ImportError:
        deps[name] = None
        print(f"  [MISSING] {name}")

missing = [k for k, v in deps.items() if v is None]
if missing:
    print(f"\nMissing: {missing}")
    sys.exit(1)
else:
    print("\nAll dependencies present.")
