#!/usr/bin/env python3

"""
Sweep experiment.py + nksynth over every GML file, for every (num-bad, num-good)
combination, profiling runtime/RAM of the nksynth solve and appending results
to a CSV as they complete.

Example:
    ./run-all.py --num-bad 1 2 3 --num-good 10 11 12 --timeout 300
"""

import argparse
import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

TIME_LINE_RE = re.compile(r"^\s*(.+?):\s*(.+?)\s*$")

VERDICT_COLORS = {"SAT": "32", "UNSAT": "31", "UNKNOWN": "34", "TIMEOUT": "33"}


def colorize(verdict):
    code = VERDICT_COLORS.get(verdict)
    return f"\033[{code}m{verdict}\033[0m" if code else verdict


def parse_elapsed(value):
    """Parse '/usr/bin/time -v' elapsed time ('m:ss.cc' or 'h:mm:ss') to seconds."""
    parts = value.split(":")
    parts = [float(p) for p in parts]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def parse_time_v_output(stderr_text):
    """Extract the metrics we care about from '/usr/bin/time -v' stderr output."""
    metrics = {
        "wall_time_s": None,
        "user_time_s": None,
        "sys_time_s": None,
        "cpu_percent": None,
        "max_rss_kb": None,
    }

    for line in stderr_text.splitlines():
        m = TIME_LINE_RE.match(line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()

        if key == "Elapsed (wall clock) time (h:mm:ss or m:ss)":
            metrics["wall_time_s"] = parse_elapsed(val)
        elif key == "User time (seconds)":
            metrics["user_time_s"] = float(val)
        elif key == "System time (seconds)":
            metrics["sys_time_s"] = float(val)
        elif key == "Percent of CPU this job got":
            metrics["cpu_percent"] = val.rstrip("%")
        elif key == "Maximum resident set size (kbytes)":
            metrics["max_rss_kb"] = int(val)

    return metrics


_current_pgid = None


def _kill_current_child(signum, frame):
    """
    If run-all.py itself is interrupted (Ctrl+C, SIGTERM) while a child is
    running, the child is in its own process group (see run_profiled) and
    would otherwise be orphaned instead of receiving the signal. Kill it
    explicitly before exiting.
    """
    if _current_pgid is not None:
        try:
            os.killpg(_current_pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    sys.exit(1)


def run_profiled(cmd, timeout, time_bin):
    """
    Run `cmd` (a list) under `/usr/bin/time -v`, in its own process group so a
    timeout can kill the whole tree. Returns (returncode, stdout, stderr,
    timed_out, wall_time_s_fallback, metrics).
    """
    global _current_pgid

    full_cmd = [time_bin, "-v"] + cmd if time_bin else cmd

    start = time.time()
    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )
    _current_pgid = proc.pid

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
    finally:
        _current_pgid = None

    wall_fallback = time.time() - start

    metrics = parse_time_v_output(stderr) if time_bin else {}
    for k in ("wall_time_s", "user_time_s", "sys_time_s", "cpu_percent", "max_rss_kb"):
        metrics.setdefault(k, None)

    return proc.returncode, stdout, stderr, timed_out, wall_fallback, metrics


def parse_node_edge_counts(stdout_text):
    """Extract 'Nodes: N' / 'Edges: N' printed by experiment.py's print_summary()."""
    num_nodes = None
    num_edges = None

    for line in stdout_text.splitlines():
        m = re.match(r"^Nodes:\s*(\d+)\s*$", line)
        if m:
            num_nodes = int(m.group(1))
            continue
        m = re.match(r"^Edges:\s*(\d+)\s*$", line)
        if m:
            num_edges = int(m.group(1))

    return num_nodes, num_edges


FINAL_VERDICTS = {"SAT", "UNSAT", "UNKNOWN"}


def load_completed(csv_path):
    """
    Return the set of (gml_file, num_bad, num_good, full, expand_indices,
    force_unsat) tuples that have a final solver verdict already recorded in
    the CSV. Rows that failed for non-solver reasons (TIMEOUT, ERROR,
    EXPERIMENT_ERROR, MISSING_OUTPUT) are left out so they get retried.
    `full`, `expand_indices`, and `force_unsat` are tracked so switching any
    of these modes doesn't skip combos solved under a different mode.
    """
    completed = set()
    if not csv_path.exists():
        return completed

    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("verdict") not in FINAL_VERDICTS:
                continue
            try:
                completed.add((
                    row["gml_file"],
                    int(row["num_bad"]),
                    int(row["num_good"]),
                    row.get("full", "True") == "True",
                    row.get("expand_indices", "True") == "True",
                    row.get("force_unsat", "False") == "True",
                ))
            except (KeyError, ValueError):
                continue

    return completed


FIELDNAMES = [
    "gml_file",
    "num_nodes",
    "num_edges",
    "num_bad",
    "num_good",
    "full",
    "expand_indices",
    "force_unsat",
    "seed",
    "experiment_returncode",
    "experiment_time_s",
    "nksynth_returncode",
    "verdict",
    "timed_out",
    "wall_time_s",
    "user_time_s",
    "sys_time_s",
    "cpu_percent",
    "max_rss_kb",
    "max_rss_mb",
    "timestamp",
]


def append_row(csv_path, row):
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def main():
    signal.signal(signal.SIGINT, _kill_current_child)
    signal.signal(signal.SIGTERM, _kill_current_child)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gml-dir", default="gml", help="Directory containing .gml files")
    parser.add_argument("--experiment-script", default="./experiment.py", help="Path to experiment.py")
    parser.add_argument("--nksynth-bin", default="../target/release/nksynth", help="Path to nksynth binary")
    parser.add_argument("--output", "-o", default="results.csv", help="CSV file to append results to")
    parser.add_argument("--num-bad", type=int, nargs="+", default=[1, 2, 3], help="num-bad values to sweep")
    parser.add_argument("--num-good", type=int, nargs="+", default=[10, 11, 12], help="num-good values to sweep")
    parser.add_argument("--seed", type=int, default=3, help="Random seed passed to experiment.py")
    parser.add_argument("--timeout", type=float, default=300, help="Timeout in seconds for the nksynth solve")
    parser.add_argument("--full", dest="full", action="store_true", default=True, help="Use nksynth --full (default)")
    parser.add_argument("--no-full", dest="full", action="store_false", help="Use nksynth without --full")
    parser.add_argument("--expand-indices", dest="expand_indices", action="store_true", help="Pass --expand-indices to experiment.py")
    parser.add_argument("--no-expand-indices", dest="expand_indices", action="store_false", default=True, help="Don't pass --expand-indices to experiment.py (default)")
    parser.add_argument("--force-unsat", action="store_true", help="Pass --force-unsat to experiment.py (selects bad paths as suffixes of good paths, to force UNSAT)")
    parser.add_argument("--force", action="store_true", help="Re-run combinations already present in the output CSV")
    args = parser.parse_args()

    gml_dir = Path(args.gml_dir)
    gml_files = sorted(gml_dir.glob("*.gml"))
    if not gml_files:
        sys.exit(f"No .gml files found in {gml_dir}")

    if not shutil.which(args.experiment_script) and not Path(args.experiment_script).exists():
        sys.exit(f"experiment script not found: {args.experiment_script}")
    if not Path(args.nksynth_bin).exists():
        sys.exit(f"nksynth binary not found: {args.nksynth_bin}")

    time_bin = shutil.which("time") or (
        "/usr/bin/time" if Path("/usr/bin/time").exists() else None
    )
    if time_bin is None:
        print("warning: /usr/bin/time not found; RAM/CPU metrics will be unavailable", file=sys.stderr)

    csv_path = Path(args.output)
    completed = set() if args.force else load_completed(csv_path)

    combos = [
        (gml_path, nb, ng)
        for gml_path in gml_files
        for nb in args.num_bad
        for ng in args.num_good
    ]

    total = len(combos)
    for i, (gml_path, num_bad, num_good) in enumerate(combos, 1):
        key = (str(gml_path), num_bad, num_good, args.full, args.expand_indices, args.force_unsat)
        if key in completed:
            print(f"[{i}/{total}] skipping (already done): {gml_path} num_bad={num_bad} num_good={num_good}")
            continue

        nksynth_path = gml_path.with_suffix(".nksynth")

        experiment_cmd = [
            args.experiment_script,
            str(gml_path),
            "--num-bad", str(num_bad),
            "--num-good", str(num_good),
            "--no-comments",
            "--no-dot",
            "--seed", str(args.seed),
        ]
        if args.expand_indices:
            experiment_cmd.append("--expand-indices")
        if args.force_unsat:
            experiment_cmd.append("--force-unsat")

        row = {
            "gml_file": str(gml_path),
            "num_nodes": None,
            "num_edges": None,
            "num_bad": num_bad,
            "num_good": num_good,
            "full": args.full,
            "expand_indices": args.expand_indices,
            "force_unsat": args.force_unsat,
            "seed": args.seed,
            "experiment_returncode": None,
            "experiment_time_s": None,
            "nksynth_returncode": None,
            "verdict": None,
            "timed_out": False,
            "wall_time_s": None,
            "user_time_s": None,
            "sys_time_s": None,
            "cpu_percent": None,
            "max_rss_kb": None,
            "max_rss_mb": None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        exp_start = time.time()
        exp_proc = subprocess.run(experiment_cmd, capture_output=True, text=True)
        row["experiment_time_s"] = round(time.time() - exp_start, 3)
        row["experiment_returncode"] = exp_proc.returncode

        num_nodes, num_edges = parse_node_edge_counts(exp_proc.stdout)
        row["num_nodes"] = num_nodes
        row["num_edges"] = num_edges

        counts_suffix = f" nodes={num_nodes} edges={num_edges}" if num_nodes is not None and num_edges is not None else ""
        print(f"[{i}/{total}] {gml_path} num_bad={num_bad} num_good={num_good}{counts_suffix}")

        if exp_proc.returncode != 0:
            print(f"  experiment.py failed (rc={exp_proc.returncode}), skipping nksynth")
            print(exp_proc.stderr, file=sys.stderr)
            row["verdict"] = "EXPERIMENT_ERROR"
            append_row(csv_path, row)
            continue

        if not nksynth_path.exists():
            print(f"  expected output not found: {nksynth_path}, skipping nksynth")
            row["verdict"] = "MISSING_OUTPUT"
            append_row(csv_path, row)
            continue

        nksynth_cmd = [str(args.nksynth_bin)]
        if args.full:
            nksynth_cmd.append("--full")
        nksynth_cmd.append(str(nksynth_path))

        returncode, stdout, stderr, timed_out, wall_fallback, metrics = run_profiled(
            nksynth_cmd, args.timeout, time_bin
        )

        row["nksynth_returncode"] = returncode
        row["timed_out"] = timed_out
        row["wall_time_s"] = metrics["wall_time_s"] if metrics["wall_time_s"] is not None else round(wall_fallback, 3)
        row["user_time_s"] = metrics["user_time_s"]
        row["sys_time_s"] = metrics["sys_time_s"]
        row["cpu_percent"] = metrics["cpu_percent"]
        row["max_rss_kb"] = metrics["max_rss_kb"]
        row["max_rss_mb"] = round(metrics["max_rss_kb"] / 1024, 2) if metrics["max_rss_kb"] else None

        if timed_out:
            row["verdict"] = "TIMEOUT"
            print(f"  {colorize('TIMEOUT')} after {args.timeout}s")
        else:
            verdict = stdout.strip().splitlines()[-1] if stdout.strip() else "ERROR"
            row["verdict"] = verdict
            print(f"  {colorize(verdict)} in {row['wall_time_s']}s, max RSS {row['max_rss_mb']} MB")
            if returncode != 0:
                print(stderr, file=sys.stderr)

        append_row(csv_path, row)


if __name__ == "__main__":
    main()
