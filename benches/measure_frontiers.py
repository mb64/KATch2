#!/usr/bin/env python3
"""Compare frontier configurations with opt-in native solver measurements."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", default="target/release/nksynth")
parser.add_argument("--samples", type=int, default=7)
parser.add_argument("--output", default="out/profiles/pigeonhole-frontier-metrics.json")
args = parser.parse_args()
modes = {
    "one_frontier": ["--no-lb-backward", "--no-lb-mincut"],
    "one_frontier_and_mincut": ["--no-lb-backward"],
    "both_frontiers": ["--no-lb-mincut"],
    "both_frontiers_and_mincut": [],
}
env = {k: v for k, v in os.environ.items() if not k.startswith("KATCH2_")}
env["KATCH2_METRICS"] = "1"
results = {str(n): {name: [] for name in modes} for n in [3, 4]}
with tempfile.TemporaryDirectory(prefix="katch2-frontiers-") as directory:
    for n in [3, 4]:
        path = Path(directory) / f"pigeon{n}.nksynth"
        tests = " + ".join(f"x{i} == 1" for i in range(n))
        path.write_text(f"hole h0\nhole h1\nassert h0; ({tests}); h1 = 1\n")
        for trial in range(args.samples):
            order = list(modes)
            shift = trial % len(order)
            order = order[shift:] + order[:shift]
            if trial % 2:
                order.reverse()
            for name in order:
                started = time.perf_counter()
                run = subprocess.run(
                    [args.binary, "--iteration-limit", "1000", *modes[name], str(path)],
                    capture_output=True, text=True, check=True, timeout=60, env=env,
                )
                elapsed = time.perf_counter() - started
                metrics = {}
                for line in run.stderr.splitlines():
                    if line.startswith("KATCH2_METRICS "):
                        fields = dict(field.split("=", 1) for field in line.split()[1:])
                        scope = fields.pop("scope")
                        assert scope not in metrics, "expected one synthesis run per input"
                        metrics[scope] = {key: int(value) for key, value in fields.items()}
                assert set(metrics) == {"sat", "native", "cegis"}, run.stderr
                verdict = run.stdout.strip()
                assert verdict == ("UNSAT" if n == 3 else "UNKNOWN"), verdict
                results[str(n)][name].append({
                    "seconds": elapsed, "verdict": verdict, "metrics": metrics,
                })
                print(f"pigeon{n} {name} trial {trial + 1}: {elapsed:.3f}s", flush=True)

for n, configurations in results.items():
    for name, rows in configurations.items():
        medians = {
            scope: {
                key: statistics.median(row["metrics"][scope].get(key, 0) for row in rows)
                for key in sorted({key for row in rows for key in row["metrics"][scope]})
            }
            for scope in ["cegis", "native", "sat"]
        }
        print(f"pigeon{n} {name}: {json.dumps(medians, sort_keys=True)}")

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    "cap": 1000, "samples": args.samples, "modes": modes, "results": results,
}, indent=2) + "\n")
