"""Run the whole test suite (every other tests/*.py), each in its own process from the repo root.

usage: python tests/run_all.py [name ...] [--list]     (names are substrings of the script name; none = everything)

Each script prints its own report and exits non-zero on failure.  `spec.py` is report-only: on prompt 3 the speculative path
may legitimately diverge from plain greedy at a bf16 exact tie (see docs/PLAN.md), so read its output rather than its exit code.
Needs the weights in models/Qwen3.5-0.8B, the golden outputs in results/ and mlx-lm (pip install -e ".[dev]"); the full run takes a few
minutes on an M5."""
import os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPORT_ONLY = {"spec.py"}

args = [a for a in sys.argv[1:] if not a.startswith("--")]
scripts = sorted(p for p in HERE.glob("*.py") if p.name != Path(__file__).name and (not args or any(a in p.stem for a in args)))
if "--list" in sys.argv:
    print("\n".join(p.name for p in scripts))
    sys.exit(0)
if not scripts:
    sys.exit(f"no test matches {args}")

env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}  # works without `pip install -e .`
results = []
for p in scripts:
    print(f"\n{'=' * 20} {p.name} {'=' * 20}", flush=True)
    t0 = time.time()
    rc = subprocess.run([sys.executable, str(p)], cwd=ROOT, env=env).returncode
    results.append((p.name, rc, time.time() - t0))

print(f"\n{'=' * 20} summary {'=' * 20}")
failed = 0
for name, rc, dt in results:
    status = "ok" if rc == 0 else ("REPORT-ONLY (rc=%d)" % rc if name in REPORT_ONLY else "FAIL (rc=%d)" % rc)
    failed += rc != 0 and name not in REPORT_ONLY
    print(f"  {name:<20} {status:<22} {dt:7.1f} s")
print(f"{len(results) - failed}/{len(results)} passed" + (f", {failed} FAILED" if failed else ""))
sys.exit(1 if failed else 0)
