"""Run the simulations behind the paper and write their data to results/.

    python reproduce/run_simulations.py --list              # show every step and what it feeds
    python reproduce/run_simulations.py                     # run them all (about two days on 2 cores)
    python reproduce/run_simulations.py --only ratchet5_rising ratchet5_falling
    python reproduce/run_simulations.py --skip-existing     # leave steps whose output is already there

Run from the code/ folder or from anywhere: commands are run with code/ as the
working folder.  The figures are then drawn by make_figures.py.
"""
import argparse
import pathlib
import subprocess
import sys
import time

CODE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE / "reproduce"))
from steps import SIM  # noqa: E402


def outputs_exist(out):
    """True when the first output named in the step is already in results/."""
    first = out.split(",")[0].split(" ")[0]
    if "{" in first:
        stem, rest = first.split("{", 1)
        first = stem + rest.split(",")[0].rstrip("}")
    return (CODE / "results" / first).exists() if "." in first else False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true")
    p.add_argument("--only", nargs="+", default=None, help="step ids to run")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    steps = [s for s in SIM if args.only is None or s[0] in args.only]
    seen = set()
    steps = [s for s in steps if not (s[0] in seen or seen.add(s[0]))]
    if args.list:
        for sid, cmd, out, note in steps:
            print(f"{sid:22s} {note}\n{'':22s} python {cmd}\n{'':22s} -> results/{out}\n")
        return
    for sid, cmd, out, note in steps:
        if args.skip_existing and outputs_exist(out):
            print(f"[skip] {sid}: results/{out} exists")
            continue
        print(f"[run ] {sid}: python {cmd}", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable] + cmd.split(), cwd=CODE, check=True)
        print(f"[done] {sid} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
